import SimpleITK as sitk
import scipy.ndimage as ndimage
import numpy as np

from soveny import visualization


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


