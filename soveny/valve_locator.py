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
import joblib  # scaler mentéséhez
from . import input as sov_input
from . import config as sov_config
from . import label as sov_label
from . import tube_BB


# ─────────────────────────────────────────────
# Feature oszlop definíciók (index → név)
# Ha a CSV-ben más a sorrend, itt kell átírni.
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    'Radius',       # 0
    'Curvature',    # 1  — extrém outlierek lehetségesek
    'Torsion',      # 2
    'Norm_X',       # 3
    'Norm_Y',       # 4
    'Norm_Z',       # 5
    'Vent_dist',    # 6  — negatív értékek lehetségesek → abs
    'Tube_len',     # 7
]
CURVATURE_IDX  = 1   # percentilis-alapú clipping
VENT_DIST_IDX  = 6   # abs() — fizikailag nem lehet negatív


def robust_preprocess(X: np.ndarray) -> np.ndarray:
    """
    Adat-szintű előfeldolgozás SKÁLÁZÁS ELŐTT:
      1. Vent_dist → abs()  (távolság nem lehet negatív)
      2. Curvature → percentilis clip a [1%, 99%] tartományra
         (az extrém outlierek, pl. 17.8 vs ~4 átlag, torzítják a StandardScalert)

    X alakja: (N, SeqLen, Features)  VAGY  (SeqLen, Features)
    """
    X = X.copy()
    orig_shape = X.shape
    if X.ndim == 3:
        N, S, F = X.shape
        X2d = X.reshape(-1, F)
    else:
        X2d = X  # (S, F)

    # 1. Vent_dist: abs
    if VENT_DIST_IDX < X2d.shape[1]:
        X2d[:, VENT_DIST_IDX] = np.abs(X2d[:, VENT_DIST_IDX])

    # 2. Curvature: percentilis clip
    if CURVATURE_IDX < X2d.shape[1]:
        lo = np.percentile(X2d[:, CURVATURE_IDX], 1)
        hi = np.percentile(X2d[:, CURVATURE_IDX], 99)
        X2d[:, CURVATURE_IDX] = np.clip(X2d[:, CURVATURE_IDX], lo, hi)

    return X2d.reshape(orig_shape)


# ─────────────────────────────────────────────
# 1. Dataset  (sample_weight támogatással)
# ─────────────────────────────────────────────
class ValveDataset(Dataset):
    def __init__(self, X_data, y_data, sample_weights=None):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)
        if sample_weights is not None:
            self.w = torch.tensor(sample_weights, dtype=torch.float32)
        else:
            self.w = torch.ones(len(X_data), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


# ─────────────────────────────────────────────
# 2. Modell architektúra
#    VÁLTOZÁSOK:
#      • num_layers: 2 → 1  (kevés adathoz kevesebb réteg kell)
#      • hidden_dim: 64 → 48 (kevesebb paraméter → kevesebb overfitting)
#      • Dropout: 0.3 → 0.4 (erősebb regularizálás)
#      • Conv1d "stem": lokális jellemzők kinyerése az LSTM előtt
# ─────────────────────────────────────────────
class ValveLocatorBiLSTM(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 48):
        super().__init__()

        # Rövid 1D-CNN előfeldolgozó: lokális mintákat tanul meg
        # (Batch, SeqLen, Features) → transpose → (Batch, Features, SeqLen) → conv → transpose vissza
        self.conv_stem = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim * 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(in_channels=input_dim * 2, out_channels=input_dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        conv_out_dim = input_dim * 2  # 16

        # Bidirekcionális LSTM
        self.lstm = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=hidden_dim,
            num_layers=1,           # 1 réteg elég 176 mintához
            batch_first=True,
            bidirectional=True,
            dropout=0.0,            # 1 rétegnél a dropout-nak nincs hatása, hagyjuk ki
        )

        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (Batch, 100, 8)
        x_conv = x.transpose(1, 2)                  # (Batch, 8, 100)
        x_conv = self.conv_stem(x_conv)              # (Batch, 16, 100)
        x_conv = x_conv.transpose(1, 2)              # (Batch, 100, 16)

        lstm_out, _ = self.lstm(x_conv)              # (Batch, 100, 96)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)                      # (Batch, 100, 1)
        return out.squeeze(-1)                       # (Batch, 100)


# ─────────────────────────────────────────────
# 3. Segédfüggvény: peak-pozíció MAE (index-ekben mérve)
# ─────────────────────────────────────────────
def peak_position_mae(predictions_logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Megadja, hogy átlagosan hány index-szel tér el a prediktált csúcs
    a valódi csúcstól (lower = better).
    """
    preds_prob = 1 / (1 + np.exp(-predictions_logits))  # sigmoid
    pred_peaks = np.argmax(preds_prob, axis=1)
    true_peaks = np.argmax(targets, axis=1)
    return float(np.mean(np.abs(pred_peaks - true_peaks)))


# ─────────────────────────────────────────────
# 4. Tanítás
# ─────────────────────────────────────────────
def train_model(
    data_path: str = "training_data.npz",
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 5e-4,
    save_path: str = "valve_locator_model.pth",
    scaler_path: str = "valve_scaler.pkl",
):
    # ── Adatok ──
    data = np.load(data_path)
    X_all = data['X']
    y_all = data['y']
    ids_all = data['ids']
    print(f"Betöltött adatok: X={X_all.shape}, y={y_all.shape}")

    # ── Robust előfeldolgozás (Vent_dist abs, Curvature clip) ──
    X_all = robust_preprocess(X_all)
    n_artery = np.sum(['artery' in str(i) for i in ids_all])
    n_aorta  = np.sum(['aorta'  in str(i) for i in ids_all])
    print(f"Típusok: {n_aorta} aorta, {n_artery} artery")

    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_all, y_all, ids_all, test_size=0.2, random_state=42
    )

    # ── Per-sample súlyok: artery esetek 1.5× súlyozása ──
    # (mert az outlierek mind artery-k, és valószínűleg alulreprezentáltak)
    train_weights = np.array([
        1.5 if 'artery' in str(i) else 1.0
        for i in ids_train
    ], dtype=np.float32)
    print(f"Train artery súly: 1.5×  |  aorta súly: 1.0×")

    # ── Skálázás ──
    N_train, seq_len, num_features = X_train.shape
    N_val = X_val.shape[0]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(N_train, seq_len, num_features)
    X_val   = scaler.transform(X_val.reshape(-1, num_features)).reshape(N_val, seq_len, num_features)

    joblib.dump(scaler, scaler_path)
    print(f"Scaler mentve: {scaler_path}")

    # ── DataLoader-ek ──
    train_loader = DataLoader(
        ValveDataset(X_train, y_train, train_weights),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        ValveDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False
    )

    # ── Modell, loss, optimizer ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Használt eszköz: {device}")

    model = ValveLocatorBiLSTM(input_dim=num_features).to(device)

    num_pos = float((y_train > 0.1).sum())
    num_neg = float((y_train <= 0.1).sum())
    pos_w   = min(num_neg / max(num_pos, 1.0), 20.0)
    print(f"pos_weight = {pos_w:.2f}")

    # Alap criterion (pos_weight nélkül — a sample weightbe visszük a súlyt)
    base_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], dtype=torch.float32).to(device),
        reduction='none'   # visszaadja az elemenként hibát, mi átlagolunk sample_weight-tel
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss    = float('inf')
    patience_limit   = 20
    patience_counter = 0

    train_losses, val_losses, val_maes = [], [], []

    for epoch in range(epochs):
        # Tanítás
        model.train()
        train_loss = 0.0
        for bX, by, bw in train_loader:
            bX, by, bw = bX.to(device), by.to(device), bw.to(device)
            optimizer.zero_grad()

            preds    = model(bX)                          # (B, 100)
            loss_el  = base_criterion(preds, by)          # (B, 100)
            loss_seq = loss_el.mean(dim=1)                # (B,) — átlag per szekvencia
            loss     = (loss_seq * bw).mean()             # súlyozott átlag

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * bX.size(0)
        train_loss /= len(train_loader.dataset)

        # Validáció
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bX, by, _ in val_loader:
                bX, by = bX.to(device), by.to(device)
                preds    = model(bX)
                loss_el  = base_criterion(preds, by)
                val_loss += loss_el.mean().item() * bX.size(0)
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

    # ── Training görbe ──
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


# ─────────────────────────────────────────────
# 5. Kiértékelés
# ─────────────────────────────────────────────
def evaluate_model(
    data_path: str  = "training_data.npz",
    model_path: str = "valve_locator_model.pth",
    scaler_path: str = "valve_scaler.pkl",
    dataset_name: str = "ImageCHD_dataset",
    sample_idx: int = 0,
):
    # ── Adatok ──
    data = np.load(data_path)
    X_all, y_all, ids_all = data['X'], data['y'], data['ids']
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_all, y_all, ids_all, test_size=0.2, random_state=42
    )

    # ── Scaler betöltése (UGYANAZ, amit tanításnál elmentettünk) ──
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler fájl nem található: {scaler_path}\n"
            "Futtasd előbb a train_model()-t, hogy legenerálódjon."
        )
    scaler = joblib.load(scaler_path)
    print(f"Scaler betöltve: {scaler_path}")

    # ── Robust előfeldolgozás (ugyanaz, mint tanításnál) ──
    X_val = robust_preprocess(X_val)

    N_val, seq_len, num_features = X_val.shape
    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(N_val, seq_len, num_features)

    # ── Modell betöltése ──
    model = ValveLocatorBiLSTM(input_dim=num_features)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # ── Teljes val-set kiértékelés (peak MAE) ──
    x_all_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    with torch.no_grad():
        all_logits = model(x_all_tensor).numpy()
    overall_mae = peak_position_mae(all_logits, y_val)
    print(f"\nValidációs set (n={N_val}) | Átlagos csúcs-pozíció MAE: {overall_mae:.2f} index")

    # ── Egy minta részletes vizsgálata ──
    ct_id = ids_val[sample_idx]
    print(f"Értékelés: {ct_id}  (sample_idx={sample_idx})")

    x_sample = torch.tensor(X_val_scaled[sample_idx:sample_idx+1], dtype=torch.float32)
    with torch.no_grad():
        logits     = model(x_sample).squeeze()
        prediction = torch.sigmoid(logits).numpy()

    y_true  = y_val[sample_idx]
    max_idx = int(np.argmax(prediction))
    true_idx = int(np.argmax(y_true))
    print(f"Prediktált csúcs: {max_idx}  |  Valódi csúcs: {true_idx}  |  Hiba: {abs(max_idx - true_idx)} idx")

    # ── Rajzolás ──
    plt.figure(figsize=(10, 5))
    plt.plot(y_true,     label='Ground Truth',  color='red',   linestyle='--', linewidth=2)
    plt.plot(prediction, label='Prediction',    color='blue',  linewidth=2)
    plt.axvline(x=max_idx,  color='blue',  linestyle=':', label=f'Pred csúcs ({max_idx})')
    plt.axvline(x=true_idx, color='red',   linestyle=':', label=f'Valódi csúcs ({true_idx})')
    plt.title(f"Pred vs GT – {ct_id} | MAE: {abs(max_idx - true_idx)} idx")
    plt.xlabel("Szekvencia index (0-99)")
    plt.ylabel("Aktiváció (0-1)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    # ── 3D vizualizáció ──
    dataset_dir, cfg = sov_config.load_config(dataset_name)
    source_dir = os.path.join(dataset_dir, "preprocessed")
    output_dir = os.path.join("output", dataset_name)

    all_csv_files   = glob.glob(os.path.join(output_dir, "**", "*resampled*.csv"), recursive=True)
    matching_files  = [f for f in all_csv_files if ct_id in f or ct_id.replace("_aorta","").replace("_artery","") in f]

    if "aorta" in ct_id:
        matching_files = [f for f in matching_files if "aorta" in f.lower() or "left" in f.lower()]
    elif "artery" in ct_id:
        matching_files = [f for f in matching_files if "artery" in f.lower() or "right" in f.lower()]

    if not matching_files:
        print(f"Nem találom a CSV fájlt a 3D vizualizációhoz: {ct_id}")
        plt.show()
        return

    df_features  = pd.read_csv(matching_files[0])
    p_z          = int(float(df_features.iloc[max_idx]['Z_index']))
    p_y          = int(float(df_features.iloc[max_idx]['Y_index']))
    p_x          = int(float(df_features.iloc[max_idx]['X_index']))
    normal_vec   = np.array([
        float(df_features.iloc[max_idx]['Norm_Z']),
        float(df_features.iloc[max_idx]['Norm_Y']),
        float(df_features.iloc[max_idx]['Norm_X']),
    ])

    ct_folder  = ct_id.replace("_aorta","").replace("_artery","")
    img_path   = os.path.join(source_dir, ct_folder.replace("_iso","_image_iso.nii.gz"))
    label_path = os.path.join(source_dir, ct_folder.replace("_iso","_label_iso.nii.gz"))

    if os.path.exists(img_path) and os.path.exists(label_path):
        print(f"CT betöltése: {img_path}")
        _, ct_arr, _, label_arr = sov_input.load_ct_and_label(img_path, label_path)
        rel_labels = sov_label.extract_labels(label_arr, cfg)

        tube_t = "aorta" if "aorta" in ct_id else "artery"
        vent_t = "left_ventricle" if tube_t == "aorta" else "right_ventricle"

        _ = tube_BB.get_cutting_plane(
            normal_vector=normal_vec,
            p_z=p_z, p_y=p_y, p_x=p_x,
            relevant_labels=rel_labels,
            ct_array=ct_arr,
            ventricle_type=vent_t,
            tube_type=tube_t,
            save_path_3d=None,
        )
    else:
        print(f"Hiányzó CT/label fájlok: {img_path}")

    plt.show()


def predict_cutting_plane_from_features(
    df_features: pd.DataFrame, 
    model_path: str = "valve_locator_model.pth", 
    scaler_path: str = "valve_scaler.pkl"
) -> tuple:
    """
    Egyetlen érpálya (pl. egy 100 hosszú DataFrame) DataFrame-jén lefuttatja a modellt 
    és visszatér a legvalószínűbb vágósík paramétereivel.
    """
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("A modell vagy a scaler nem található. Kérlek futtasd le a tanítást előbb!")
        
    # Ezek az oszlopok kellettek a tanításhoz is (a main-ből kiindulva)
    feature_cols = ['Tubeness', 'Radius', 'L1', 'L2', 'L3', 'Norm_Z', 'Norm_Y', 'Norm_X']
    
    # 1. Bemenet összeállítása (1, 100, 8) alakú numpy tömb
    X_sample = df_features[feature_cols].values
    X_sample = X_sample[np.newaxis, :, :] # Batch dimenzió hozzáadása
    
    # 2. Skálázás
    import joblib
    scaler = joblib.load(scaler_path)
    
    # Robust előfeldolgozás ugyanolyan indexekkel, ahogy a tanításnál volt
    X_sample = robust_preprocess(X_sample)
    
    N, seq_len, num_features = X_sample.shape
    X_scaled = scaler.transform(X_sample.reshape(-1, num_features)).reshape(N, seq_len, num_features)
    
    # 3. Modell betöltése (CPU-ra)
    model = ValveLocatorBiLSTM(input_dim=num_features)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # 4. Predikció előállítása
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor).squeeze()
        prediction = torch.sigmoid(logits).numpy()
        
    # 5. Legjobb index kiválasztása
    max_idx = int(np.argmax(prediction))
    
    # 6. Értékek kiolvasása az eredeti DataFrame-ből (a koordinátákhoz)
    p_z = int(float(df_features.iloc[max_idx]['Z_index']))
    p_y = int(float(df_features.iloc[max_idx]['Y_index']))
    p_x = int(float(df_features.iloc[max_idx]['X_index']))
    
    norm_z = float(df_features.iloc[max_idx]['Norm_Z'])
    norm_y = float(df_features.iloc[max_idx]['Norm_Y'])
    norm_x = float(df_features.iloc[max_idx]['Norm_X'])
    
    normal_vector = np.array([norm_z, norm_y, norm_x])
    
    return p_z, p_y, p_x, normal_vector, max_idx

def diagnose_model(
    data_path: str   = "training_data.npz",
    model_path: str  = "valve_locator_model.pth",
    scaler_path: str = "valve_scaler.pkl",
):
    """
    A teljes validációs seten végigmegy és részletes hibaelemzést ad:
    - Per-sample csúcs MAE táblázat (rendezve hiba szerint)
    - Hibaeloszlás hisztogram
    - A 4 legjobb és 4 legrosszabb eset predikciós görbéje
    """
    data = np.load(data_path)
    X_all, y_all, ids_all = data['X'], data['y'], data['ids']
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_all, y_all, ids_all, test_size=0.2, random_state=42
    )

    scaler = joblib.load(scaler_path)
    N_val, seq_len, num_features = X_val.shape

    # ── Robust előfeldolgozás ──
    X_val = robust_preprocess(X_val)

    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(N_val, seq_len, num_features)

    model = ValveLocatorBiLSTM(input_dim=num_features)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        all_logits = model(torch.tensor(X_val_scaled, dtype=torch.float32)).numpy()

    all_probs   = 1 / (1 + np.exp(-all_logits))
    pred_peaks  = np.argmax(all_probs,  axis=1)
    true_peaks  = np.argmax(y_val,      axis=1)
    errors      = np.abs(pred_peaks - true_peaks)

    # ── Táblázat ──
    df = pd.DataFrame({
        'id':        ids_val,
        'true_peak': true_peaks,
        'pred_peak': pred_peaks,
        'error_idx': errors,
    }).sort_values('error_idx')

    print("\n── Validációs eredmények (hiba szerint rendezve) ──")
    print(df.to_string(index=False))
    print(f"\nÁtlag MAE : {errors.mean():.2f} idx")
    print(f"Medián MAE: {np.median(errors):.1f} idx")
    print(f"≤5 idx    : {(errors <= 5).mean()*100:.1f}%")
    print(f"≤10 idx   : {(errors <= 10).mean()*100:.1f}%")
    print(f">20 idx   : {(errors > 20).mean()*100:.1f}%  ← outlierek")

    # ── Hisztogram ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(errors, bins=20, color='steelblue', edgecolor='white')
    axes[0].axvline(errors.mean(),   color='red',    linestyle='--', label=f'Átlag={errors.mean():.1f}')
    axes[0].axvline(np.median(errors), color='orange', linestyle='--', label=f'Medián={np.median(errors):.1f}')
    axes[0].set_title('Csúcs-pozíció hiba eloszlása')
    axes[0].set_xlabel('|pred_peak − true_peak| (index)')
    axes[0].set_ylabel('Darab')
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    # ── Pred vs GT görbék: top-4 legjobb + top-4 legrosszabb ──
    sorted_idx = np.argsort(errors)
    best_4  = sorted_idx[:4]
    worst_4 = sorted_idx[-4:]

    axes[1].barh(
        range(len(df)), df['error_idx'].values,
        color=['#2ecc71' if e <= 5 else '#e67e22' if e <= 15 else '#e74c3c'
               for e in df['error_idx'].values]
    )
    axes[1].set_yticks(range(len(df)))
    axes[1].set_yticklabels([f"{row['id']} (T:{row['true_peak']})" for _, row in df.iterrows()], fontsize=7)
    axes[1].set_title('Per-sample hiba')
    axes[1].set_xlabel('Hiba (index)')
    axes[1].axvline(10, color='red', linestyle=':', label='10 idx határ')
    axes[1].legend(); axes[1].grid(True, alpha=0.4, axis='x')

    plt.tight_layout()
    plt.savefig("diagnosis_overview.png", dpi=120)

    # ── Görbe grid ──
    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 7))
    fig2.suptitle("Legjobb 4 és Legrosszabb 4 predikció", fontsize=13)

    for col, idx in enumerate(best_4):
        ax = axes2[0, col]
        ax.plot(y_val[idx],       color='red',  linestyle='--', label='GT')
        ax.plot(all_probs[idx],   color='blue', label='Pred')
        ax.axvline(pred_peaks[idx], color='blue',  linestyle=':')
        ax.axvline(true_peaks[idx], color='red',   linestyle=':')
        ax.set_title(f"✓ {ids_val[idx]}\nhiba={errors[idx]} idx", fontsize=8)
        ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Legjobb 4')

    for col, idx in enumerate(worst_4):
        ax = axes2[1, col]
        ax.plot(y_val[idx],       color='red',  linestyle='--', label='GT')
        ax.plot(all_probs[idx],   color='blue', label='Pred')
        ax.axvline(pred_peaks[idx], color='blue',  linestyle=':')
        ax.axvline(true_peaks[idx], color='red',   linestyle=':')
        ax.set_title(f"✗ {ids_val[idx]}\nhiba={errors[idx]} idx", fontsize=8)
        ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Legrosszabb 4')
        if col == 3: ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("diagnosis_curves.png", dpi=120)

    # ── Outlier mélyelemzés: feature görbék a legrosszabb 4 esethez ──
    FEATURE_NAMES = [
        'Radius', 'Curvature', 'Torsion',
        'Norm_X', 'Norm_Y', 'Norm_Z',
        'Vent_dist', 'Tube_len'
    ]

    print("\n── Outlier mélyelemzés ──")
    for rank, idx in enumerate(worst_4):
        ct     = ids_val[idx]
        raw_x  = X_val[idx]        # robust-preprocessed, de NEM skálázott feature értékek
        true_p = true_peaks[idx]
        pred_p = pred_peaks[idx]
        err    = errors[idx]

        # Gyanús minták ellenőrzése
        edge_warn = ""
        if true_p < 10:
            edge_warn = "⚠  GT csúcs a ELEJÉN van (<10 idx) — cső végén van a szelep?"
        elif true_p > 90:
            edge_warn = "⚠  GT csúcs a VÉGÉN van (>90 idx) — cső végén van a szelep?"

        gt_at_pred  = y_val[idx, pred_p]
        low_gt_note = " ⚠ hamis csúcs gyanú" if gt_at_pred < 0.3 else ""

        print(f"\n[#{rank+1}] {ct} | hiba={err} idx | true={true_p}, pred={pred_p}")
        if edge_warn:
            print(f"  {edge_warn}")
        print(f"  GT érték a pred csúcsnál: {gt_at_pred:.3f}{low_gt_note}")

        print(f"  {'Feature':<12} {'@true_peak':>12} {'@pred_peak':>12} {'diff':>10}")
        n_feat = raw_x.shape[1]
        for fi in range(n_feat):
            fname  = FEATURE_NAMES[fi] if fi < len(FEATURE_NAMES) else f'f{fi}'
            v_true = raw_x[true_p, fi]
            v_pred = raw_x[pred_p, fi]
            print(f"  {fname:<12} {v_true:>12.4f} {v_pred:>12.4f} {v_pred - v_true:>10.4f}")

        # Feature görbék mentése
        fig3, ax3 = plt.subplots(n_feat, 1, figsize=(12, 2 * n_feat), sharex=True)
        fig3.suptitle(f"Feature görbék: {ct}  (true={true_p}, pred={pred_p}, hiba={err})", fontsize=11)
        for fi in range(n_feat):
            fname = FEATURE_NAMES[fi] if fi < len(FEATURE_NAMES) else f'f{fi}'
            ax3[fi].plot(raw_x[:, fi], color='gray', linewidth=1)
            ax3[fi].axvline(true_p, color='red',  linestyle='--', linewidth=1.5,
                            label='True csúcs' if fi == 0 else "")
            ax3[fi].axvline(pred_p, color='blue', linestyle=':',  linewidth=1.5,
                            label='Pred csúcs' if fi == 0 else "")
            ax3[fi].set_ylabel(fname, fontsize=8)
            ax3[fi].grid(True, alpha=0.3)
        ax3[0].legend(fontsize=7, loc='upper right')
        ax3[-1].set_xlabel("Szekvencia index")
        plt.tight_layout()
        safe_ct = ct.replace("/", "_").replace("\\", "_")
        plt.savefig(f"outlier_{safe_ct}.png", dpi=100)
        print(f"  → Mentve: outlier_{safe_ct}.png")
        plt.close(fig3)

    plt.show()


if __name__ == "__main__":
    train_model(data_path="training_data.npz", epochs=100)
    diagnose_model(data_path="training_data.npz")