import os
import numpy as np
import scipy.ndimage as ndimage
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import SimpleITK as sitk

from . import filter
from . import output
from . import visualization

def get_cutting_indices(sorted_indices, scores_on_skel, eigenvectors_on_skel, sz_orig, sy_orig, sx_orig, z_min: int = 0, y_min: int = 0, x_min: int = 0, threshold_og: float = 0.9) -> tuple[tuple[int, int, int], np.ndarray]:
    cut_index = -1
    threshold = threshold_og

    last_score = 0.9

    for idx in sorted_indices:
        current_score = scores_on_skel[idx]
            
        if current_score - last_score > threshold:
            cut_index = idx
            break
        
        last_score = current_score


    if cut_index != -1:
        p_z, p_y, p_x = int(sz_orig[cut_index]), int(sy_orig[cut_index]), int(sx_orig[cut_index])
        normal_vector: np.ndarray = eigenvectors_on_skel[cut_index]
        print(f"Vágási pont megtalálva: Z={p_z}, Y={p_y}, X={p_x}, Normál vektor: {normal_vector}")
        return (p_z, p_y, p_x), normal_vector
    else:
        print("Hiba: Nem találtam olyan pontot, ami átlépi a küszöböt!")
        return (-1, -1, -1), np.array([0.0, 0.0, 0.0])


def get_cutting_plane(ventricle_label: np.ndarray, tube_label: np.ndarray, ct_array: np.ndarray = None, ct_image: sitk.Image = None, out_dir: str = None, ventricle_type: str = None, tube_type: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Keres egy vágósíkot a kamra és a cső (pl. aorta) között, amely elválasztja a kettőt.
    
    Args:
        ventricle_label: a kamra maszkja (pl. bal kamra)
        tube_label: a cső maszkja (pl. aorta)
        ct_array: A CT kép a vizualizációhoz (opcionális)
        ct_image: A CT kép a vizualizációhoz (opcionális)
        out_dir: A kimeneti könyvtár elérési útvonala (opcionális)
        ventricle_type: A kamra típusa (pl. bal kamra)
        tube_type: A cső típusa (pl. aorta)
    Returns:
        visual_plane_mask: A sík megjelenítéshez használt maszkja (néhány voxel vastag).
        half_space_mask: Egy bináris maszk, amellyel vágni lehet a kamrát 
                         (True a sík "alatt/mögött", False az eldobandó részen).
    """
    
    print("A cső ténylegesen csőszerű részének meghatározása")
    ventricle_dilated = ndimage.binary_dilation(ventricle_label, iterations=80)
    tubular_tube = ventricle_dilated & tube_label
    
    # Csak a vizualizációhoz
    if ct_array is None:
        ct_array = np.zeros_like(ventricle_label, dtype=np.float32)
    if ct_image is None:
        ct_image = sitk.GetImageFromArray(ct_array)
    if tube_type is None:
        tube_type = "tube"
    if ventricle_type is None:
        ventricle_type = "ventricle"
        
    out_dir += f"_{tube_type}" if tube_type else ""
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        
    z_idx, y_idx, x_idx = np.where(tubular_tube)
    y_center = y_idx.mean().astype(int)
    if tube_type == "artery":
        y_center = y_idx.min() + 30 
    og_tube_label = tube_label.copy()

    #vizualizációs ellenőrzés
    print("A cső ténylegesen csőszerű része")
    visualization.plot_slice_with_labels(ct_array, {tube_type : og_tube_label, ventricle_type : ventricle_label, 'tubular_tube': tubular_tube}, axis='y', slice_idx=y_center)

    
    #kimentés ellenőrzéshez
    output.save_array_as_image(tubular_tube.astype(np.uint8), ct_image, os.path.join(out_dir, "tubular_tube_check.nii.gz"))
    
    tube_label = tubular_tube
    unio = ventricle_label | tube_label
    smoothed = ndimage.binary_dilation(unio, iterations=20)
    smoothed = ndimage.binary_erosion(smoothed, iterations=20)
        
    # vizualizációs ellenőrzéshez
    print("Unio = kamra és az igazi cső, ez simítva:")
    visualization.plot_slice_with_labels(ct_array, {tube_type : og_tube_label, ventricle_type : ventricle_label, 'smoothed_union': smoothed}, axis='y', slice_idx=y_center)

    # kimentés ellenőrzéshez
    output.save_array_as_image(smoothed.astype(np.uint8), ct_image, os.path.join(out_dir, "smoothed_union_check.nii.gz"))
    
    skeleton = skeletonize(smoothed)
    
    # kimentés ellenőrzéshez
    big_skeleton = ndimage.binary_dilation(skeleton, iterations=2)
    output.save_array_as_image(big_skeleton.astype(np.uint8), ct_image, os.path.join(out_dir, "skeleton_check.nii.gz"))
    
    # Region of Interest meghatározása az átmenet kereséséhez
    tube_dilated = ndimage.binary_dilation(tube_label, iterations=15)
    ventricle_dilated = ndimage.binary_dilation(ventricle_label, iterations=80)

    tube_parts = np.logical_and(tube_dilated, ventricle_dilated)

    # vizualizációs ellenőrzéshez
    print("A csőszerű rész amin keresünk: a kamra csőszerű része és a cső csőszerű részének dilatált metszete")
    visualization.plot_slice_with_labels(ct_array, {tube_type : og_tube_label, ventricle_type : ventricle_label, 'tube_parts': tube_parts}, axis='y', slice_idx=y_center)
    
    interesting_region = np.logical_and(tube_parts, np.logical_or(ventricle_label, tube_label))

    print("A csőszerű részt finomítjuk, a kamra és a cső csőszerű részének eredeti uniójára")
    visualization.plot_slice_with_labels(ct_array, {tube_type : og_tube_label, ventricle_type : ventricle_label, 'interesting_region': interesting_region}, axis='y', slice_idx=y_center)

    interesting_region = ndimage.binary_erosion(interesting_region, iterations=4)
    interesting_region = ndimage.binary_dilation(interesting_region, iterations=4)

    # vizualizációs ellenőrzéshez
    print("Simítjuk az előzőt")
    visualization.plot_slice_with_labels(ct_array, {tube_type : og_tube_label, ventricle_type : ventricle_label, 'interesting_region': interesting_region}, axis='y', slice_idx=y_center)
    # kimentés ellenőrzéshez
    output.save_array_as_image(interesting_region.astype(np.uint8), ct_image, os.path.join(out_dir, "interesting_region_check.nii.gz"))
    
    interesting_skeleton = skeleton & interesting_region
    big_interesting_skeleton = ndimage.binary_dilation(interesting_skeleton, iterations=1)
    
    # kimentés ellenőrzéshez
    output.save_array_as_image(big_interesting_skeleton.astype(np.uint8), ct_image, os.path.join(out_dir, "interesting_skeleton_check.nii.gz"))
    
    # A binary_mask lehet az 'interesting_region', vagy az aorta/kamra eredeti uniója is.
    # Fontos, hogy ez az a vastag maszk legyen, aminek a falától a távolságot mérjük!
    distance_map = distance_transform_edt(interesting_region.astype(bool))
    radii_in_voxels = distance_map[interesting_skeleton == 1]

    if len(radii_in_voxels) > 0:
        min_r = np.min(radii_in_voxels)
        max_r = np.max(radii_in_voxels)
        mean_r = np.mean(radii_in_voxels)
        
        print(f"\nAz ér/kamra szakasz sugarai a skeleton mentén (voxelben):")
        print(f"  Minimum sugár: {min_r:.2f}")
        print(f"  Maximum sugár: {max_r:.2f}")
        print(f"  Átlagos sugár: {mean_r:.2f}")
        
        # Ebből már automatikusan generálhatod a Frangi szűrő szigmáit!
        # Pl. a minimum és maximum közé generálunk 5 lépcsőt:
        dynamic_sigmas = np.linspace(mean_r, max_r, num=5).tolist()
        print(f"  --> Automatikusan javasolt szigmák a szűrőhöz: {dynamic_sigmas}")
    else:
        print("Nem található skeleton pont a távolságok kinyeréséhez.")
    
    z_idx, y_idx, x_idx = np.nonzero(smoothed)
            
    z_min: int = max(0, z_idx.min())
    z_max: int = min(smoothed.shape[0], z_idx.max() + 1)
        
    y_min: int = max(0, y_idx.min())
    y_max: int = min(smoothed.shape[1], y_idx.max() + 1)
        
    x_min: int = max(0, x_idx.min())
    x_max: int = min(smoothed.shape[2], x_idx.max() + 1)
        
        # CT kivágása
    smoothed_cropped = smoothed[z_min:z_max, y_min:y_max, x_min:x_max]
    skeleton_cropped = skeleton[z_min:z_max, y_min:y_max, x_min:x_max]

    print(f"Kivágott doboz mérete: Z:{z_max-z_min}, Y:{y_max-y_min}, X:{x_max-x_min}")
    tubeness_result = filter.multiscale_tubeness_3d(
            image=smoothed_cropped.astype(np.float32), 
            sigmas=dynamic_sigmas, 
            skeleton=skeleton_cropped, 
            alpha=0.5, 
            beta=0.5
        )
    # tubeness kimentése ellenőrzéshez
    tubeness_full_check = np.zeros_like(ct_array, dtype=np.float32)
    tubeness_full_check[z_min:z_max, y_min:y_max, x_min:x_max] = tubeness_result['scores']
    output.save_array_as_image(tubeness_full_check, ct_image, os.path.join(out_dir, "tubeness_scores_check.nii.gz"))
    
    scores_cropped = tubeness_result['scores']
    eigenvectors_cropped = tubeness_result['eigenvectors']

    z_skel, y_skel, x_skel = np.nonzero(skeleton_cropped)

    # Kinyerjük az értékeket pontosan a skeleton pontjain
    scores_on_skel = scores_cropped[z_skel, y_skel, x_skel]
    eigenvectors_on_skel = eigenvectors_cropped[z_skel, y_skel, x_skel]

    # Sorba rendezzük Z szerint növekvő sorrendbe (Lábtól a Fej felé)
    sorted_indices = np.argsort(z_skel)

    p_z = -1
    threshold = 0.9
    
    while p_z == -1:
        (p_z, p_y, p_x), normal_vector = get_cutting_indices(
            sorted_indices=sorted_indices,
            scores_on_skel=scores_on_skel,
            eigenvectors_on_skel=eigenvectors_on_skel,
            sz_orig=z_skel,
            sy_orig=y_skel,
            sx_orig=x_skel,
            z_min=z_min,
            y_min=y_min,
            x_min=x_min,
            threshold_og=threshold
        )
        threshold *= 0.95  # Ha nem találunk pontot, akkor engedünk a küszöbön, hogy ne maradjunk pont nélkül
    
    # --- SÍK GENERÁLÁSA ---
    n_z, n_y, n_x = normal_vector[0], normal_vector[1], normal_vector[2]

    # Eredeti, nem vágott koordinátarendszerbe való visszaállítás
    p_z += z_min
    p_y += y_min
    p_x += x_min
        
    # Kicsit eltoljuk a vágási pontot, hogy a kamra biztonságban legyen
    p_z -= int(abs(n_z) * 1)
    p_y -= int(abs(n_y) * 1)
    p_x -= int(abs(n_x) * 1)

    shape = ventricle_label.shape
    Z, Y, X = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
        
    plane_equation = n_z * (Z - p_z) + n_y * (Y - p_y) + n_x * (X - p_x)

    visual_plane_mask = np.abs(plane_equation) <= 2.0

    # Irány korrigálása, hogy a Z tengely (fej felé) vágjon lefelé
    if n_z > 0:
        plane_equation = -plane_equation

    half_space_mask = plane_equation > 0 
    # vizualizációs ellenőrzéshez
    print("A vágási sík a cső és a kamra között")
    visualization.plot_slice_with_labels(ct_array, {tube_type : og_tube_label, ventricle_type : ventricle_label, 'cutting_plane': visual_plane_mask}, axis='y', slice_idx=y_center)
    # kimnetés ellenőrzéshez
    out_name = f"cutting_plane_check_{tube_type}.nii.gz" if tube_type else "cutting_plane_check.nii.gz"
    output.save_array_as_image(visual_plane_mask.astype(np.uint8), ct_image, os.path.join(out_dir, out_name))
    
    return visual_plane_mask, half_space_mask