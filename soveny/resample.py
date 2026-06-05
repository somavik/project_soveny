from pathlib import Path
from turtle import pd

import SimpleITK as sitk
import scipy.ndimage as ndimage
import numpy as np

from soveny import visualization

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resample_to_isotropic(ct_image: sitk.Image, label_image: sitk.Image, ct_order=3, starting_spacing : tuple = (1.0, 1.0, 1.0)) -> tuple[sitk.Image, sitk.Image]:

    # sitk.DICOMOrientImageFilter() használatával a képet standard orientációba helyezhetjük
    # 1. Készíts egy orientáló szűrőt
    orient_filter = sitk.DICOMOrientImageFilter()

    # 2. Állítsd be a kívánt orientációt (pl. LPS, RAS, RIA stb.)
    # Az elérhető irányok az ITK konvenciói szerinti 3 betűből állnak.
    orient_filter.SetDesiredCoordinateOrientation("RAS") 

    # 3. Futtasd le a szűrőt az eredeti sitk.Image(k)en
    reoriented_ct_image = orient_filter.Execute(ct_image)
    reoriented_label_image = orient_filter.Execute(label_image)

    # Opcionális: kiírhatod az új irányokat, hogy ellenőrizd
    # Pl.: "LPS" feltehetően a standard DICOM/NIfTI megfelelő lesz
    # print("Új orientáció (CT):", sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(reoriented_ct_image.GetDirection()))

    # 4. Kérd le az átirányított numpy tömböket a plotoláshoz vagy további feldolgozáshoz
    reoriented_ct_array: np.ndarray = sitk.GetArrayFromImage(reoriented_ct_image)
    reoriented_label_array: np.ndarray = sitk.GetArrayFromImage(reoriented_label_image)

    # 1. Definiáljuk az eredeti és a cél felbontást NUMPY (Z, Y, X) sorrendben!
    # Az excel alapján: X=0.30078, Y=0.30078, Z=0.75
    z, y, x = starting_spacing
    
    original_spacing_zyx = (z, y, x) 

    # Legyen a cél egy tökéletes izotróp 1x1x1 mm-es rács
    target_spacing_zyx = (1.0, 1.0, 1.0) 

    # 2. Kiszámoljuk a nagyítási/kicsinyítési arányokat tengelyenként
    # Ha pl. Z-ben 0.75 volt és 1.0-ra megyünk, akkor a zoom faktor 0.75 (zsugorítjuk a mátrixot)
    zoom_factors = [orig / targ for orig, targ in zip(original_spacing_zyx, target_spacing_zyx)]
    #print(f"Numpy Zoom faktorok (Z, Y, X): {zoom_factors}")

    # 3. CT átmintavételezése LINEÁRIS interpolációval (order=1)
    #print("CT átmintavételezése folyamatban...")
    resampled_ct_array = ndimage.zoom(reoriented_ct_array, zoom=zoom_factors, order=3)

    # 4. Maszk (Label) átmintavételezése NEAREST NEIGHBOR interpolációval (order=0)
    # Ez KRITIKUS, mert a címkék diszkrét értékek (0, 1, 2). Ha ide lineárisat tennél,
    # a határvonalakon 0.5 meg 1.3 értékű "címkék" keletkeznének, ami tönkretenné a maszkot!
    #print("Címkék átmintavételezése folyamatban...")
    resampled_label_array = ndimage.zoom(reoriented_label_array, zoom=zoom_factors, order=0)

    #print(f"Új izotróp CT shape: {resampled_ct_array.shape}")


    # 5. Eredmény ellenőrzése
    visualization.plot_slice_with_labels(resampled_ct_array, resampled_label_array, axis="y")
    visualization.plot_slice_with_labels(resampled_ct_array, resampled_label_array, axis="x")
    visualization.plot_slice_with_labels(resampled_ct_array, resampled_label_array, axis="z")

    resampled_ct_image = sitk.GetImageFromArray(resampled_ct_array)
    resampled_ct_image.SetSpacing((1.0, 1.0, 1.0))
    resampled_ct_image.SetOrigin(reoriented_ct_image.GetOrigin())
    resampled_ct_image.SetDirection(reoriented_ct_image.GetDirection())
    
    resampled_label_image = sitk.GetImageFromArray(resampled_label_array)
    resampled_label_image.SetSpacing((1.0, 1.0, 1.0))
    resampled_label_image.SetOrigin(reoriented_label_image.GetOrigin())
    resampled_label_image.SetDirection(reoriented_label_image.GetDirection())

    return resampled_ct_image, resampled_label_image


import SimpleITK as sitk
from torch import flip

def resample_to_isotropic_ras(input_path, output_path, is_label=False,
                               original_spacing_xyz=None):
    
    
    img = sitk.ReadImage(input_path)
    if not is_label:
        img = sitk.Cast(img, sitk.sitkFloat32)
        img = img - 1024  # HU normalizálás (csak CT-re, labelre nem)

    if original_spacing_xyz is not None:
        img.SetSpacing(original_spacing_xyz)

    # KULCSLÉPÉS: direction kényszerítése LPS-re az orientálás előtt
    # (DICOM-ból jövő képeknél az alap LPS, nem RAS)
    img.SetDirection((1, 0, 0,
                      0, 1, 0,
                      0, 0, 1))  # ideiglenesen identity-re állítjuk

    # RAS orientálás
    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("RAS")
    img_ras = orient.Execute(img)

    # Ellenőrzés
    print("Direction after RAS:", img_ras.GetDirection())

    orig_spacing = img_ras.GetSpacing()
    orig_size    = img_ras.GetSize()
    new_size = [int(round(sz * sp)) for sz, sp in zip(orig_size, orig_spacing)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing([1.0, 1.0, 1.0])
    resample.SetSize(new_size)
    resample.SetOutputDirection(img_ras.GetDirection())
    resample.SetOutputOrigin(img_ras.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(-1024 if not is_label else 0)
    resample.SetInterpolator(
        sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    )
    resampled = resample.Execute(img_ras)
    
    sitk.WriteImage(resampled, output_path)
    return resampled

# 2. Módosítsuk a függvényt, hogy fogadja a már betöltött DataFrame-et
def get_spacing_from_excel(image_path: str, df: pd.DataFrame) -> tuple:
    patient_id_str = os.path.basename(image_path).split('_')[1].replace("ct", "")
    patient_idx = int(patient_id_str) - 1000
    print(f"\n--- Feldolgozás: Beteg {patient_id_str} (Keresett azonosító: {patient_idx}) ---")
    
    # A 1-es indexű oszlop (B) tartalmazza a betegszámot
    patient_row = df[df[1] == patient_idx].iloc[0]
        
    # 9-es oszlop (J): Spacing X, 10-es (K): Spacing Y, 12-es (M): Spacing Z
    sp_x_str = str(patient_row[9]).replace(',', '.')
    sp_y_str = str(patient_row[10]).replace(',', '.')
    sp_z_str = str(patient_row[12]).replace(',', '.')
        
    correct_spacing = (float(sp_x_str), float(sp_y_str), float(sp_z_str))
    print(f"  > Kiolvasott spacing: {correct_spacing}")
    
    return correct_spacing

def resample_dataset_to_isotropic_ras(resume_from: str = None):
    dataset_dir = r"ImageCHD_dataset\\ImageCHD_dataset"

    search_pattern = os.path.join(dataset_dir, "*_image.nii.gz")
    image_paths = sorted(glob.glob(search_pattern))
        
    print(f"Összesen {len(image_paths)} CT fájl található.")

    resampled_dir = r"ImageCHD_dataset\\resampled"
    # Kimeneti könyvtár létrehozása, ha nem létezik
    os.makedirs(resampled_dir, exist_ok=True)
    
    print("Excel betöltése...")
    excel_path = r"ImageCHD_dataset\ImageCHD_dataset\imagechd_dataset_image_info.xlsx"
    df_info = pd.read_excel(excel_path, header=None)
    
    
    skip = resume_from is not None
    for img_path in image_paths:
        if skip:
            if resume_from in os.path.basename(img_path):
                skip = False
            else:
                continue

        label_path = img_path.replace("_image", "_label")
        if os.path.exists(label_path):
            selected = Path(img_path)
            # Kinyerjük a fájlnevét a kiterjesztések nélkül (nii, nii.gz), és eltávolítjuk az "_image" részt ha benne van
            file_name = selected.name.split('.')[0].replace('_image', '')
            
            resampled_dir_patient = os.path.join(resampled_dir, file_name)
            os.makedirs(resampled_dir_patient, exist_ok=True)
            
            ct_output_path = os.path.join(resampled_dir_patient,  os.path.basename(img_path).replace("_image", "_iso_image"))
            label_output_path = os.path.join(resampled_dir_patient, os.path.basename(label_path).replace("_label", "_iso_label"))
            
            # A JAVÍTOTT PRINT SOROK (pl. tabulátorral behúzva)
            print(f"\tct: {os.path.basename(img_path)}")
            print(f"\tcimke: {os.path.basename(label_path)}")
            
            print(f"kimenet ct: {os.path.basename(ct_output_path)}")
            print(f"kimenet cimke: {os.path.basename(label_output_path)}")
            
            # 3. Itt átadjuk a már betöltött df_info-t!
            image_spacing = get_spacing_from_excel(img_path, df_info)
            
            ct = resample_to_isotropic_ras(
                img_path,
                ct_output_path,
                is_label=False,
                original_spacing_xyz=image_spacing
                )

            # Label
            label = resample_to_isotropic_ras(
                label_path,
                label_output_path,
                is_label=True,
                original_spacing_xyz=image_spacing
                )
        else:
            print(f"Nem található label fájl: {label_path}")
    
    
def fix_header_and_orientation(input_path, output_path, is_label=False, original_spacing_xyz=None):
    img = sitk.ReadImage(input_path)
    
    # Kép típusának beállítása és HU normalizálás (ImageCHD sajátosság)
    if not is_label:
        img = sitk.Cast(img, sitk.sitkFloat32)
        img = img - 1024  
    else:
        # A címkék maradhatnak kicsi, egész számos típusok (helytakarékos)
        img = sitk.Cast(img, sitk.sitkUInt8)

    # 1. Spacing beírása a fejlécbe
    if original_spacing_xyz is not None:
        img.SetSpacing(original_spacing_xyz)

    # 2. KULCSLÉPÉS: direction kényszerítése LPS-re az orientálás előtt
    img.SetDirection((1, 0, 0,
                      0, 1, 0,
                      0, 0, 1))

    # 3. RAS orientálás (Forgatás a memóriában)
    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("RAS")
    img_ras = orient.Execute(img)

    # Nincs interpoláció (ResampleImageFilter)! 
    # A képtömb (array) mérete és értékei érintetlenek maradnak, 
    # csak a térbeli elhelyezkedése lett helyreállítva.
    
    sitk.WriteImage(img_ras, output_path)
    return img_ras

def fix_dataset_headers(resume_from: str = None):
    dataset_dir = r"ImageCHD_dataset\ImageCHD_dataset"
    search_pattern = os.path.join(dataset_dir, "*_image.nii.gz")
    image_paths = sorted(glob.glob(search_pattern))
        
    print(f"Összesen {len(image_paths)} CT fájl található.")

    # Új kimeneti mappa neve, hogy ne keveredjen az iso-val
    fixed_dir = r"ImageCHD_dataset\fixed_headers" 
    os.makedirs(fixed_dir, exist_ok=True)
    
    print("Excel betöltése...")
    excel_path = r"ImageCHD_dataset\ImageCHD_dataset\imagechd_dataset_image_info.xlsx"
    df_info = pd.read_excel(excel_path, header=None)
    
    skip = resume_from is not None
    for img_path in image_paths:
        if skip:
            if resume_from in os.path.basename(img_path):
                skip = False
            else:
                continue

        label_path = img_path.replace("_image", "_label")
        if os.path.exists(label_path):
            selected = Path(img_path)
            file_name = selected.name.split('.')[0].replace('_image', '')
            
            fixed_dir_patient = os.path.join(fixed_dir, file_name)
            os.makedirs(fixed_dir_patient, exist_ok=True)
            
            # Nevezzük el beszédesen a kimenetet
            ct_output_path = os.path.join(fixed_dir_patient, os.path.basename(img_path).replace("_image", "_fixed_image"))
            label_output_path = os.path.join(fixed_dir_patient, os.path.basename(label_path).replace("_label", "_fixed_label"))
            
            print(f"\tct: {os.path.basename(img_path)}")
            print(f"\tcimke: {os.path.basename(label_path)}")
            print(f"\tkimenet ct: {os.path.basename(ct_output_path)}")
            
            image_spacing = get_spacing_from_excel(img_path, df_info)
            
            # CT feldolgozása
            fix_header_and_orientation(
                img_path,
                ct_output_path,
                is_label=False,
                original_spacing_xyz=image_spacing
            )

            # Label feldolgozása
            fix_header_and_orientation(
                label_path,
                label_output_path,
                is_label=True,
                original_spacing_xyz=image_spacing
            )
        else:
            print(f"Nem található label fájl: {label_path}")