import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import glob
import pandas as pd
import joblib 
from . import input as sov_input
from . import config as sov_config
from . import label as sov_label
from . import tube_BB



FEATURE_NAMES = [
    'Z_index',  
    'Y_index', 
    'X_index',  
    'Tubeness', 
    'Radius',   
    'L1',       
    'L2',       
    'L3',       
    'Norm_Z',   
    'Norm_Y',   
    'Norm_X',   
]

# Dataset  
class ValveDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class _ResidualDilatedBlock(nn.Module):
    """
    Egy dilated Conv1d blokk reziduális kapcsolattal:
      x -> Conv(dilation=d) -> GELU -> Dropout -> Conv(1×1) -> + x -> -> GroupNorm
    """
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        pad = dilation # hogy lehetséges legyen a reziduális kapcsolat (input és output ugyanakkora hosszú legyen), így jön ki a matek
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation), # kontextus bővítése felbontás vesztés nélkül, memória hatékonyan. Pl dilation=2, kernel=3, első elem: [p][p][x1]_[x2]
            nn.GELU(), # nemlinearitás
            nn.Dropout(dropout), # regularizáció
            nn.Conv1d(channels, channels, kernel_size=1), # csatorna keverés
        )
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels) # normalizáció a stabilitásért

    def forward(self, x):
        return self.norm(x + self.block(x))  # + reziduálizáció


class ValveLocatorCNN(nn.Module):
    """
    Seq2Seq dilated CNN:
      Bemenet : (batch mérete (B), 100, input_dim)   — (B, Sequence_length, Features)
      Kimenet : (B, 100)              — logits minden pozícióhoz

    Receptive field számítás (kernel=3, dilations=[1,2,4,8,16]):
      RF = 1 + 2 * sum(dilations) = 1 + 2*(1+2+4+8+16) = 63 pont
    """
    def __init__(self, input_dim: int = 11, channels: int = 32, dropout: float = 0.2):
        super().__init__()

        # Bemenet vetítése a belső channels-dimenzióba
        # Később több feature adása a hálónak mint 11. PL.több skála szerinti csőszerűség és normálvektor értékek, vagy akár CT intenzitás a fővonalon. 
        # Így több csatornára projektálni mint 32. 
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=1),
            nn.GELU(),
        )

        # 5 dilated blokk, növekvő látótávolság
        self.blocks = nn.Sequential(
            _ResidualDilatedBlock(channels, dilation=1,  dropout=dropout),
            _ResidualDilatedBlock(channels, dilation=2,  dropout=dropout),
            _ResidualDilatedBlock(channels, dilation=4,  dropout=dropout),
            _ResidualDilatedBlock(channels, dilation=8,  dropout=dropout),
            _ResidualDilatedBlock(channels, dilation=16, dropout=dropout),
        )

        # Kimeneti fej: channels -> 1 logit pozíciónként
        self.head = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 100, 11)
        x = x.transpose(1, 2)      # (B, 11, 100)  — CNN channels-first
        x = self.input_proj(x)     # (B, 32, 100)
        x = self.blocks(x)         # (B, 32, 100)
        x = self.head(x)           # (B, 1,  100)
        return x.squeeze(1)        # (B, 100)



# Segédfüggvény: peak-pozíció MAE (index-ekben mérve)
def peak_position_mae(predictions_logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Megadja, hogy átlagosan hány index-szel tér el a prediktált csúcs
    a valódi csúcstól (lower = better).
    """
    pred_peaks = np.argmax(predictions_logits, axis=1)
    true_peaks = np.argmax(targets, axis=1)
    return float(np.mean(np.abs(pred_peaks - true_peaks)))


# Tanítás
def train_model(
    data_path: str = "training_data.npz",       # x: (30, 100, 11) y: (30, 100)
    epochs: int = 100,                          # hányszor fusson le a teljes adathalmaz a hálón
    batch_size: int = 16,                       # egyszerre feldolgozott minták száma
    lr: float = 5e-4,                           # tanulási ráta
    save_path: str = "valve_locator_model.pth", # betanított modell
    scaler_path: str = "valve_scaler.pkl",      # a tanításnál használt scaler, hogy ugyanazt használhassuk a kiértékelésnél is
):
    # Adatok 
    data = np.load(data_path)
    X_all = data['X']
    y_all = data['y']
    ids_all = data['ids'] # kiértékelésnél hasznos
    print(f"Betöltött adatok: X={X_all.shape}, y={y_all.shape} ids={ids_all.shape}")

    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_all, y_all, ids_all, test_size=0.2, random_state=42
    )

    # Skálázás
    N_train, seq_len, num_features = X_train.shape
    N_val = X_val.shape[0]

    scaler = StandardScaler() # normál eloszlásúvá tesszük a feature-öket
    # Kilapítjuk, transzformáljuk, majd visszaalakítjuk
    X_train = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(N_train, seq_len, num_features)   # a scaler megtanulja a statisztikát és alkalmazza a tanító adatokra
    X_val   = scaler.transform(X_val.reshape(-1, num_features)).reshape(N_val, seq_len, num_features)           # a scaler csak alkalmazza a tanult statisztikát a validációs adatokra

    # Elmentjük a scalert, hogy a kiértékelésnél ugyanazt használhassuk
    joblib.dump(scaler, scaler_path)
    print(f"Scaler mentve: {scaler_path}")

    # DataLoader-ek
    train_loader = DataLoader(
        ValveDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True # a tanító adatok keverése minden epoch elején, hogy a háló ne tanulja meg a sorrendet és jobban általánosítson
    )
    val_loader = DataLoader(
        ValveDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False
    )

    # Device, ha van GPU, használjuk azt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Használt eszköz: {device}")

    # Modell, korábban definiált
    model = ValveLocatorCNN(input_dim=num_features).to(device)
    print(f"Modell: ValveLocatorCNN  |  Paraméterek: {sum(p.numel() for p in model.parameters()):,}")

    # Kompenzálás
    num_pos = float((y_train > 0.1).sum())              # azon pontok száma, ahol elfogadható a vágás (kevés) - pozitív
    num_neg = float((y_train <= 0.1).sum())             # azon pontok száma, ahol nem jó a vágás (sok) - negatív
    pos_w   = min(num_neg / max(num_pos, 1.0), 20.0)    # hányszor több negatív példa van mint pozitív
    print(f"pos_weight = {pos_w:.2f}")
    # -> Pl pos_weight=10
    # Egy jó vágás hely eltévesztése 10-szer súlyosabb, mint egy rossz vágás hely bemondása

    # Loss function
    base_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], dtype=torch.float32).to(device)
    )

    # Optimizer és scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)     # súlyok regularizációja
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # tanulási ráta csökkentése

    
    best_val_loss    = float('inf')
    patience_limit   = 20
    patience_counter = 0

    train_losses, val_losses, val_maes = [], [], []

    for epoch in range(epochs):
       # Tanítás
        model.train()
        train_loss = 0.0
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)       # a batch adatait átrakjuk a GPU-ra (ha van)
            optimizer.zero_grad()                       # gradiens nullázása, hogy ne keveredjen a korábbi batch-ek gradiensével

            preds = model(bX)                           # (16, 100) — tippek a batch-re
            loss  = base_criterion(preds, by)           # a teljes batch átlagos hibája

            loss.backward()                             # gradiens számítása a veszteség függvény szerint
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # gradiens klippelés, hogy ne legyenek túl nagy gradiens értékek
            optimizer.step()                            # súlyok frissítése a gradiens alapján
            train_loss += loss.item() * bX.size(0)      # összegyűjtjük a batch veszteségeket, súlyozva a batch méretével
        train_loss /= len(train_loader.dataset)         # átlagos veszteség a teljes tanító adathalmazon

        # Validáció
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bX, by in val_loader:
                bX, by = bX.to(device), by.to(device)
                preds = model(bX)
                loss  = base_criterion(preds, by)     # Egyetlen szám: a validációs batch átlagos hibája
                
                val_loss += loss.item() * bX.size(0)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(by.cpu().numpy())
        val_loss /= len(val_loader.dataset)

        all_preds   = np.concatenate(all_preds,   axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        val_mae     = peak_position_mae(all_preds, all_targets)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Peak MAE: {val_mae:.1f} idx"
            )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"\nEarly stopping a(z) {epoch+1}. epochnál.")
            break

    print(f"\nTanítás kész! Legjobb modell: {save_path}  (Val Loss: {best_val_loss:.4f})")

    # Tanítás görbélye
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train loss')
    ax1.plot(val_losses,   label='Val loss')
    ax1.set_title('BCE Loss'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(True)

    ax2.plot(val_maes, color='orange', label='Val Peak MAE')
    ax2.set_title('Csúcs-pozíció MAE (index)'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    plt.show(block=False)

    return model


def evaluate_model(
    data_path: str  = "training_data.npz",
    model_path: str = "valve_locator_model.pth",
    scaler_path: str = "valve_scaler.pkl",
    dataset_name: str = "ImageCHD_dataset",
    sample_idx: int = 0,
):
    #Adatok betöltése és felosztása
    data = np.load(data_path)
    X_train, X_val, y_train, y_val, _, ids_val = train_test_split(
        data['X'], data['y'], data['ids'], test_size=0.2, random_state=42
    )

    # Scaler betöltése és adatok skálázása
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler fájl nem található: {scaler_path}\nFuttasd előbb a train_model()-t!")
    
    scaler = joblib.load(scaler_path)
    N_val, seq_len, num_features = X_val.shape
    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(N_val, seq_len, num_features)

    # Modell betöltése és teljes körű inferencia
    model = ValveLocatorCNN(input_dim=num_features)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        all_logits = model(torch.tensor(X_val_scaled, dtype=torch.float32)).numpy()
    
    overall_mae = peak_position_mae(all_logits, y_val)
    print(f"\nValidációs set (n={N_val}) | Átlagos csúcs-pozíció MAE: {overall_mae:.2f} index")

    # Egyetlen minta vizsgálata
    ct_id = ids_val[sample_idx]
    
    logits = all_logits[sample_idx]
    prediction = 1 / (1 + np.exp(-logits)) # Sigmoid, hogy 0-1 közé szorítsuk a predikciót, így könnyebben értelmezhető 
    y_true = y_val[sample_idx]

    max_idx = int(np.argmax(prediction))
    true_idx = int(np.argmax(y_true))
    error = abs(max_idx - true_idx)
    
    print(f"Értékelés: {ct_id}  (sample_idx={sample_idx})")
    print(f"Prediktált csúcs: {max_idx}  |  Valódi csúcs: {true_idx}  |  Hiba: {error} idx")

    # Rajzolás
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Ground Truth', color='red', linestyle='--', linewidth=2)
    plt.plot(prediction, label='Prediction', color='blue', linewidth=2)
    plt.axvline(x=max_idx, color='blue', linestyle=':', label=f'Pred csúcs ({max_idx})')
    plt.axvline(x=true_idx, color='red', linestyle=':', label=f'Valódi csúcs ({true_idx})')
    plt.title(f"Pred vs GT – {ct_id} | MAE: {error} idx")
    plt.xlabel("Szekvencia index (0-99)")
    plt.ylabel("Aktiváció (0-1)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_model(data_path="training_data_lv_aorta.npz", epochs=100)
    evaluate_model(data_path="training_data_lv_aorta.npz", model_path="valve_locator_model.pth", scaler_path="valve_scaler.pkl", dataset_name="lidc", sample_idx=2)
