"""Orchestrator script: ties modules together into a simple pipeline.

Usage (example):
    python -m soveny.main --auto
"""
import argparse
from ast import arg
import os
from sympy import arg
import numpy as np

from . import tube_BB
from . import config
from . import input
from . import label
from . import ventircles_BB
from . import visualization
from . import septum_BB
from . import filter
from . import output

import glob

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def create_training_dataset(labels_csv_path: str, data_dir: str, output_file: str = "training_data.npz"):
    """
    Összeállítja a X (jellemzők) és y (célváltozó) tenzorokat a neurális háló számára.
    """
    # 1. Beolvassuk a címkéket
    df_labels = pd.read_csv(labels_csv_path)
    
    # Ezeket a jellemzőket (oszlopokat) fogjuk használni a tanításhoz
    # A Z, Y, X koordinátákat NE tegyük bele a tanítóadatba, mert a háló ne a térbeli 
    # pozíciót tanulja meg, hanem a CSŐ alakjának változását!
    feature_cols = ['Tubeness', 'Radius', 'L1', 'L2', 'L3', 'Norm_Z', 'Norm_Y', 'Norm_X']
    
    X_list = []
    y_list = []
    valid_ids = []
    
    # Keresünk minden resampled csv fájlt
    all_csv_files = glob.glob(os.path.join(data_dir, "**", "*resampled*.csv"), recursive=True)
    
    sigma = 5.0 # A Gauss-görbe szélessége 
    indices = np.arange(100) # 0-tól 99-ig

    for index, row in df_labels.iterrows():
        ct_id = str(row['CT_ID'])
        optimal_idx = int(row['Optimal_Index'])
        
        # Megpróbáljuk megtalálni a CT_ID-hoz tartozó fájlt
        # Ez a te pontos elnevezési logikádtól függhet, de pl. a ct_1001_iso_aorta alapján:
        matching_files = [f for f in all_csv_files if ct_id in f or ct_id.replace("_aorta", "").replace("_artery", "") in f]
        
        # Mivel egy mappában lehet aorta és artery is, pontosítunk:
        if "aorta" in ct_id:
            matching_files = [f for f in matching_files if "aorta" in f.lower() or "left" in f.lower()]
        elif "artery" in ct_id:
            matching_files = [f for f in matching_files if "artery" in f.lower() or "right" in f.lower()]

        if not matching_files:
            print(f"Nem találom a fájlt ehhez: {ct_id}")
            continue
            
        file_path = matching_files[0]
        
        # 2. Jellemzők (Features) beolvasása
        df_features = pd.read_csv(file_path)
        
        if len(df_features) != 100:
            print(f"Hiba: A {ct_id} fájl nem 100 hosszú! Kihagyás.")
            continue
            
        # Kiválasztjuk a szükséges oszlopokat és Numpy tömbbé alakítjuk (100, 8)
        X_array = df_features[feature_cols].values
        
        # 3. Célváltozó (Target) legenerálása Gauss-görbével (100,)
        # Figyelem: Itt a sor indexet (0-99) használjuk távolságként, NEM a Z koordinátát!
        y_array = np.exp(-((indices - optimal_idx) ** 2) / (2 * sigma ** 2))
        
        # Nagyon pici értékeket levágjuk nullára a tiszta matek miatt
        y_array[y_array < 0.01] = 0.0
        
        X_list.append(X_array)
        y_list.append(y_array)
        valid_ids.append(ct_id)
        
    # Listák átalakítása 3D Numpy Tenzorokká
    X = np.array(X_list) # Alakja: (Minták_száma, 100, 8)
    y = np.array(y_list) # Alakja: (Minták_száma, 100)
    
    print(f"\nSikeresen feldolgozva: {len(X)} minta.")
    print(f"X (Bemenet) alakja: {X.shape}")
    print(f"y (Célváltozó) alakja: {y.shape}")
    
    # 4. Mentés
    np.savez(output_file, X=X, y=y, ids=valid_ids)
    print(f"Adatok elmentve: {output_file}")

def run_labeling_pipeline(data_dir: str, source_dir: str, labels_file: str = "master_labels.csv", resume_from: str = None, single_case: str = None):
    """
    Végigmegy a megadott könyvtárban található összes resampled CSV fájlon,
    kiplotolja a jellemzőket, és bekéri a felhasználótól az ideális vágási indexet.
    """
    
    # 1. Eddig címkézett adatok beolvasása (hogy ott folytassuk, ahol abbahagytuk)
    labeled_cases = set()
    if os.path.exists(labels_file):
        df_existing = pd.read_csv(labels_file)
        if not df_existing.empty:
            labeled_cases = set(df_existing['CT_ID'].astype(str).tolist())
    else:
        # Ha még nincs ilyen fájl, létrehozzuk a fejléccel
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.write("CT_ID,Optimal_Index\n")
            
    print(f"Eddig címkézve: {len(labeled_cases)} db CT.")

    # 2. Megkeressük az összes CSV fájlt a mappában (és almappáiban)
    # Feltételezzük, hogy a fájlok neve tartalmazza a "resampled" szót
    search_pattern = os.path.join(data_dir, "**", "*resampled*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print("Nem találtam feldolgozott CSV fájlokat a megadott mappában!")
        return

    def show_2d_dashboard(df_data, current_id):
        # 4. Vizualizáció készítése (Dashboard)
        f, axs = plt.subplots(4, 1, figsize=(10, 12))
        axs[1].sharex(axs[0])
        axs[2].sharex(axs[0])
        if hasattr(f.canvas.manager, 'set_window_title'):
            f.canvas.manager.set_window_title(f'CT ID: {current_id}')
            
        x_ax = df_data.index
        axs[0].plot(x_ax, df_data['Tubeness'], marker='.', color='blue')
        axs[0].set_title('Csőszerűség (Tubeness) - Keresd a hirtelen zuhanást!')
        axs[0].set_ylabel('Tubeness')
        axs[0].grid(True)
        
        axs[1].plot(x_ax, df_data['Radius'], marker='.', color='green')
        axs[1].set_title('Középvonal sugara (Radius)')
        axs[1].set_ylabel('Radius (voxel)')
        axs[1].grid(True)
        
        axs[2].plot(x_ax, df_data['Norm_Z'], label='Norm_Z', color='red', alpha=0.7)
        axs[2].plot(x_ax, df_data['Norm_Y'], label='Norm_Y', color='orange', alpha=0.7)
        axs[2].plot(x_ax, df_data['Norm_X'], label='Norm_X', color='purple', alpha=0.7)
        axs[2].set_title('Normálvektor komponensek - Keresd a szétesést/kifordulást!')
        axs[2].set_xlabel('Index (0-99)')
        axs[2].set_ylabel('Vektor érték')
        axs[2].legend()
        axs[2].grid(True)

        axs[3].plot(df_data['Z_index'], x_ax, marker='.', color='magenta')
        axs[3].set_title('Z koordináta az út mentén')
        axs[3].set_ylabel('Z koordináta (Z_index)')
        axs[3].set_xlabel('Út index (0-99)')
        axs[3].grid(True)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1) # Kicsit várunk, hogy a GUI betöltsön
        return f

    # 3. Iterálás a fájlokon
    skip_mode = resume_from is not None
    from pathlib import Path
    for file_path in sorted(csv_files):
        path_obj = Path(file_path)
        ct_folder = path_obj.parents[1].name  # pl. ct_1001_iso
        vessel_folder = path_obj.parents[0].name  # pl. aorta_cutting_plane
        vessel_name = vessel_folder.split('_')[0] # aorta
        ct_id = f"{ct_folder}_{vessel_name}" # pl. ct_1001_iso_aorta
        
        if single_case:
            if single_case not in ct_id:
                continue
            # Ha egyedi esetet nezunk, nem ugorjuk at, meg ha mar cimkezve van is
        else:
            if skip_mode:
                if resume_from in ct_id:
                    skip_mode = False
                else:
                    continue

            if ct_id in labeled_cases:
                continue # Ezt már címkézted, ugrunk a következőre
            
        print(f"\n--- Következő eset: {ct_id} ---")
        df = pd.read_csv(file_path)
        
        if df.empty or len(df) < 100:
            print(f"Figyelem: A {ct_id} adatai hiányosak. Ugrás...")
            continue

        # 4. Vizualizáció készítése (Dashboard)
        fig = show_2d_dashboard(df, ct_id)
        
        # 5. Felhasználói input bekérése
        import builtins
        while True:
            user_input = builtins.input(f"[{ct_id}] Kérem az optimális vágási indexet (0-99), 's' (skip), vagy 'q' (kilépés): ").strip().lower()
            
            if user_input == 'q':
                print("Címkézés megszakítva. Állapot mentve.")
                plt.close('all')
                return
            elif user_input == 's':
                print("Eset átugorva.")
                break # Kilép a while loopból, megy a következő for iterációra
                
            try:
                optimal_idx = int(user_input)
                if 0 <= optimal_idx <= 99:
                    
                    # Vágósík generálása és megjelenítése
                    try:
                        p_z = int(float(df.at[optimal_idx, 'Z_index']))
                        p_y = int(float(df.at[optimal_idx, 'Y_index']))
                        p_x = int(float(df.at[optimal_idx, 'X_index']))
                        norm_z = df.at[optimal_idx, 'Norm_Z']
                        norm_y = df.at[optimal_idx, 'Norm_Y']
                        norm_x = df.at[optimal_idx, 'Norm_X']
                        normal_vec = np.array([norm_z, norm_y, norm_x])
                        
                        # Megkeressük az eredeti CT és label fájlt a 3D plotoláshoz
                        img_path = os.path.join(source_dir, ct_folder.replace("_iso", "_image_iso.nii.gz"))
                        label_path = os.path.join(source_dir, ct_folder.replace("_iso", "_label_iso.nii.gz"))
                        
                        if os.path.exists(img_path) and os.path.exists(label_path):
                            print("3D vizualizáció betöltése...")
                            plt.close(fig) # Bezárjuk a 2D plotot
                            _, ct_arr, _, label_arr = input.load_ct_and_label(img_path, label_path)
                            dataset_name = os.path.basename(os.path.normpath(data_dir))
                            dataset_cfg = config.load_config(dataset_name)[1]
                            rel_labels = label.extract_labels(label_arr, dataset_cfg)
                            
                            # Kiderítjük hogy aorta vagy arteria (mappa útvonalából)
                            tube_t = "aorta" if "aorta" in file_path else "artery"
                            vent_t = "left_ventricle" if tube_t == "aorta" else "right_ventricle"
                            
                            tube_BB.get_cutting_plane(
                                normal_vector=normal_vec, 
                                p_z=p_z, p_y=p_y, p_x=p_x, 
                                relevant_labels=rel_labels,
                                ct_array=ct_arr, 
                                ventricle_type=vent_t, 
                                tube_type=tube_t,
                                save_path_3d=None
                            )
                        else:
                            print(f"Nem sikerült betölteni a 3D nézethez a CT-t és a címkét: {img_path}")
                            
                    except Exception as e:
                        print(f"Hiba a 3D megjelenítés során: {e}")

                    # Második megerősítés kérése a 3D nézet után
                    confirm = builtins.input("Elfogadod ezt a vágást? (y/n): ").strip().lower()
                    if confirm == 'y':
                        # 6. Eredmény mentése (hozzáfűzés a fájlhoz vagy frissítés)
                        if ct_id in labeled_cases:
                            df_existing = pd.read_csv(labels_file)
                            df_existing.loc[df_existing['CT_ID'] == ct_id, 'Optimal_Index'] = optimal_idx
                            df_existing.to_csv(labels_file, index=False)
                        else:
                            with open(labels_file, 'a', encoding='utf-8') as f:
                                f.write(f"{ct_id},{optimal_idx}\n")
                        labeled_cases.add(ct_id)
                        print(f"-> {ct_id} elmentve az indexszel: {optimal_idx}")
                        break
                    else:
                        print("Vágás elutasítva. Kérlek válassz másikat! A diagramok újratöltése...")
                        fig = show_2d_dashboard(df, ct_id)
                else:
                    print("Kérlek 0 és 99 közötti számot adj meg!")
            except ValueError:
                print("Érvénytelen formátum! Számot, 's'-t vagy 'q'-t kérek.")
                
        # Grafikon bezárása a következő előtt
        plt.close(fig)

    print("\nGratulálok! Az összes elérhető CT felvétel címkézve lett!")

def process_single_dataset(image_path: str, label_path: str, dataset_name: str, cfg: dict):
    from pathlib import Path
    print(f"\n--- Feldolgozás alatt: {os.path.basename(image_path)} ---")
    ct_image, ct_array, _, label_array = input.load_ct_and_label(image_path, label_path)
    
    output_dir = output.derive_output_dir(image_path, dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    relevant_labels_dic = label.extract_labels(label_array, cfg)
    
    roi_mask = relevant_labels_dic['left_ventricle'] | relevant_labels_dic['right_ventricle']
    
    cropped_ct_array, cropped_relevant_labels_dic = ventircles_BB.crop_to_roi(
        ct_array, 
        relevant_labels_dic, 
        roi_mask
    )
    
    #visualization.plot_slice_with_labels(cropped_ct_array, cropped_relevant_labels_dic, axis='z', save_path=os.path.join(output_dir, 'ventricles_overlay.png'))
    
    septum_mask = septum_BB.get_septum_by_distance(
        cropped_relevant_labels_dic['left_ventricle'],
        cropped_relevant_labels_dic['right_ventricle'],
        max_distance_mm=12
    )
    
    #visualization.plot_slice_with_labels(cropped_ct_array, {'septum': septum_mask,}, axis='z', save_path=os.path.join(output_dir, 'septum_overlay.png'))

    _ = tube_BB.get_cutting_features(
        ventricle_label=relevant_labels_dic['left_ventricle'],
        tube_label=relevant_labels_dic['aorta'],
        ct_array=ct_array,
        ct_image=ct_image,
        out_dir=os.path.join(output_dir, "aorta_cutting_plane"),
        tube_type='aorta',
        ventricle_type='left_ventricle'
    )
    
    _ = tube_BB.get_cutting_features(
        ventricle_label=relevant_labels_dic['right_ventricle'],
        tube_label=relevant_labels_dic['artery'],
        ct_array=ct_array,
        ct_image=ct_image,
        out_dir=os.path.join(output_dir, "artery_cutting_plane"),
        tube_type='artery',
        ventricle_type='right_ventricle'
    )

    print(f"Befejezve: {os.path.basename(image_path)}")

def process_all_datasets(dataset_name: str, dataset_dir: str, cfg: dict, resume_from: str = None):
    search_pattern = os.path.join(dataset_dir, "*_image_iso.nii.gz")
    image_paths = sorted(glob.glob(search_pattern))
    
    print(f"Összesen {len(image_paths)} CT fájl található.")
    
    skip = resume_from is not None
    for img_path in image_paths:
        if skip:
            if resume_from in os.path.basename(img_path):
                skip = False
            else:
                continue

        label_path = img_path.replace("_image_iso", "_label_iso")
        if os.path.exists(label_path):
            process_single_dataset(img_path, label_path, dataset_name, cfg)
        else:
            print(f"Nem található label fájl: {label_path}")

def main(dataset_name: str = 'ImageCHD_dataset'):
    
    parser = argparse.ArgumentParser(description='Run the Soveny pipeline on a selected image and label.')
    parser.add_argument('--dataset', type=str, default=dataset_name, help=f'Name of the dataset to load (default: {dataset_name})')
    # Argumentum meghagyása, de most alapértelmezetten mindent feldolgozunk
    parser.add_argument('--single', action='store_true', help='Csak egy darab kép manuális kiválasztása')
    parser.add_argument('--resume', type=str, default=None, help='Kép neve ahonnan folytatni szeretnéd a feldolgozást (pl. ct_1030)')
    parser.add_argument('--case', type=str, default=None, help='Bizonyos eset (pl. ct_1037) újracímkézése külön')
    parser.add_argument('--labeling', action='store_true', help='Kézi címkéző interfész indítása az output mappán')
    parser.add_argument('--create_dataset', action='store_true', help='Tanító adathalmaz létrehozása a master_labels.csv alapján')
    args = parser.parse_args()

    if args.create_dataset:
        output_dir = os.path.join("output", args.dataset)
        print(f"Tanító adathalmaz létrehozása ebből a mappából: {output_dir}")
        create_training_dataset(labels_csv_path="master_labels.csv", data_dir=output_dir, output_file="training_data.npz")
        return

    if args.labeling:
        dataset_dir, cfg = config.load_config(args.dataset)
        source_dir = os.path.join(dataset_dir, "preprocessed")
        output_dir = os.path.join("output", args.dataset)
        print(f"Címkéző pipeline indítása: {output_dir}")
        run_labeling_pipeline(data_dir=output_dir, source_dir=source_dir, labels_file="master_labels.csv", resume_from=args.resume, single_case=args.case)
        return

    dataset_dir, cfg = config.load_config(args.dataset)
    dataset_dir = os.path.join(dataset_dir, "preprocessed")  # Resampled könyvtár használata most az ImageCHD_dataset-ben, ahol már izotrópra van resample-elve a CT és a label is.
    
    if args.single:
        image_path, label_path = input.get_input_paths(dataset_dir)
        process_single_dataset(image_path, label_path, args.dataset, cfg)
    else:
        # A kérésnek megfelelően ráengedjük az összesre a meghívást:
        process_all_datasets(args.dataset, dataset_dir, cfg, args.resume)
    
    
if __name__ == '__main__':
    main()
