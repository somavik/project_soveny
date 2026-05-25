import os
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import SimpleITK as sitk
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


from soveny import filter
from soveny import output
from soveny import visualization

def get_shortest_path_between_farest_points(skeleton : np.ndarray) -> np.ndarray:
    z,y,x = np.nonzero(skeleton)
    skeleton_neighbors = {}

    for i in range(len(z)):
        z_coord, y_coord, x_coord = z[i], y[i], x[i]
        neighbors = []
        
        for j in range(len(z)):
            if i == j:
                continue
            z_neighbor, y_neighbor, x_neighbor = z[j], y[j], x[j]
            dx = abs(x_neighbor - x_coord)
            dy = abs(y_neighbor - y_coord)
            dz = abs(z_neighbor - z_coord)
            if dx <= 1 and dy <= 1 and dz <= 1:
                neighbors.append((z_neighbor, y_neighbor, x_neighbor))

        skeleton_neighbors[(z_coord, y_coord, x_coord)] = neighbors

    farest_points = ()
    longest_distance = 0.0
    longest_distance_idx = (0, 0)

    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            point1 = (z[i], y[i], x[i])
            point2 = (z[j], y[j], x[j])
            distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)
            
            if distance > longest_distance:
                longest_distance = distance
                longest_distance_idx = (i, j)

    farest_points = (longest_distance, (z[longest_distance_idx[0]], y[longest_distance_idx[0]], x[longest_distance_idx[0]]), (z[longest_distance_idx[1]], y[longest_distance_idx[1]], x[longest_distance_idx[1]]))

    shortest_path = find_shortest_path(farest_points[1], farest_points[2], skeleton_neighbors)
    return shortest_path

def find_shortest_path(start_point : tuple, end_point : tuple, skeleton_neighbors : dict) -> np.ndarray:
    from collections import deque
    visited = set()
    queue = deque([(start_point, [start_point])])  # (current_point, path_to_current)
    
    while queue:
        current_point, path = queue.popleft()
        
        if current_point == end_point:
            length = len(path)
            
            return np.array(path)
        
        if current_point in visited:
            continue
        
        visited.add(current_point)
        
        for neighbor in skeleton_neighbors.get(current_point, []):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    
    return np.array([])  # Nincs út a két pont között


def get_cutting_indices(sorted_indices, scores_on_skel, eigenvectors_on_skel, sz_orig, sy_orig, sx_orig, z_min: int = 0, y_min: int = 0, x_min: int = 0, threshold_og: float = 0.9) -> tuple[tuple[int, int, int], np.ndarray]:
    cut_index = -1
    threshold = threshold_og

    last_score = 0.1

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

def get_max_tubeness_indices(scores_on_skel, eigenvectors_on_skel, sz, sy, sx) -> tuple[tuple[int, int, int], np.ndarray]:
    if len(scores_on_skel) == 0:
        return (-1, -1, -1), np.array([0.0, 0.0, 0.0])
    
    max_idx = int(np.argmax(scores_on_skel))
    max_score = scores_on_skel[max_idx]
    
    p_z, p_y, p_x = int(sz[max_idx]), int(sy[max_idx]), int(sx[max_idx])
    normal_vector: np.ndarray = eigenvectors_on_skel[max_idx]
    
    print(f"Megtalálva: Z: {p_z}, {max_score}")
    print(f"Vágási pont megtalálva: Z={p_z}, Y={p_y}, X={p_x}, Normál vektor: {normal_vector}")
    return (p_z, p_y, p_x), normal_vector


def resample_dataframe(original_df: pd.DataFrame) -> pd.DataFrame:  
    FIXED_LENGTH = 100
    current_length = len(original_df)
    
    # Ha valamiért teljesen üres lenne az út, adjunk vissza üres DataFrame-et
    if current_length == 0:
        return original_df
        
    # Létrehozunk egy "idő tengelyt" 0-tól 1-ig a jelenlegi pontokhoz
    t_current = np.linspace(0, 1, current_length)
    
    # Létrehozunk egy új "idő tengelyt" 0-tól 1-ig pontosan 100 ponthoz
    t_new = np.linspace(0, 1, FIXED_LENGTH)
    
    # Egy üres szótár az interpolált adatoknak
    resampled_data = {}
    
    # Végigmegyünk a DataFrame minden oszlopán
    for col in original_df.columns:
        # Létrehozzuk az interpolációs függvényt az adott oszlopra
        # kind='linear' tökéletes, de lehet 'cubic' is a finomabb görbületekhez
        f_interp = interp1d(t_current, original_df[col].values, kind='linear')
        
        # Kiszámoljuk az új, 100 darab értéket
        resampled_data[col] = f_interp(t_new)
        
    # Létrehozzuk az új, immár pontosan 100 soros táblázatot
    df_resampled = pd.DataFrame(resampled_data)
    
    # Visszaalakítjuk az indexeket egész számokká, mert az interpoláció tizedestörtet csinált belőlük
    df_resampled['Z_index'] = np.round(df_resampled['Z_index']).astype(int)
    df_resampled['Y_index'] = np.round(df_resampled['Y_index']).astype(int)
    df_resampled['X_index'] = np.round(df_resampled['X_index']).astype(int)

    return df_resampled

def get_smoothed(scale: int, ventricle_label: np.ndarray, tube_label: np.ndarray) -> np.ndarray:
    distance = 1 * scale
    iterations = 5
    
    smoothed_ventricle = ndimage.binary_dilation(ventricle_label, iterations=iterations)
    smoothed_ventricle = ndimage.binary_erosion(smoothed_ventricle, iterations=iterations)
          
    dist_to_ventricle = distance_transform_edt(~ventricle_label)
    
    tubular_tube = (dist_to_ventricle < distance) & tube_label
    
    unio = smoothed_ventricle | tubular_tube
    
    smoothed = ndimage.binary_dilation(unio, iterations=3)
    smoothed = ndimage.binary_erosion(smoothed, iterations=3)
    
    return smoothed.astype(np.uint8)

def get_cutting_features(ventricle_label: np.ndarray, tube_label: np.ndarray, ct_array: np.ndarray = None, ct_image: sitk.Image = None, out_dir: str = None, ventricle_type: str = None, tube_type: str = None) -> pd.DataFrame:
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
        df.DataFrame: A vágási jellemzőket tartalmazó DataFrame.
    """ 
    scale = 0.5
    while True:
        smoothed = get_smoothed(scale=scale, ventricle_label=ventricle_label, tube_label=tube_label)
        skeleton = skeletonize(smoothed)
        
        # Szigetek keresése 26-os szomszédsággal
        struct_26 = ndimage.generate_binary_structure(3, 3)
        labeled_skeleton, num_features = ndimage.label(skeleton, structure=struct_26)
        
        if num_features > 0:
            # Legnagyobb sziget megkeresése (a bincount 0. eleme a háttér, azt kihagyjuk)
            sizes = np.bincount(labeled_skeleton.flat)[1:]
            largest_label = np.argmax(sizes) + 1
            main_skeleton = (labeled_skeleton == largest_label)
            
            main_path = get_shortest_path_between_farest_points(main_skeleton)
            
            main_path_mask = np.zeros_like(main_skeleton, dtype=np.uint8)
            if len(main_path) > 0:
                main_path_mask[tuple(main_path.T)] = 1            
            # a main path nem csak a kamrában, de a csőben is megy, akkor az jó
            if tube_label[main_path_mask.astype(bool)].sum() > 3:
                break
            else:
                scale += 1.0
        else:
            scale += 1.5
            
    # Region of Interest meghatározása az átmenet kereséséhez, a fő vonal felső 10%-a
    cut_len = int(len(main_path) * 0.1) 
    bottleneck_path = main_path[:cut_len]

    distance_map = distance_transform_edt(smoothed.astype(bool))
    if len(bottleneck_path) > 0:
        radii_in_voxels = distance_map[tuple(bottleneck_path.T)] # át kell alakítani koordinátákká
    else:
        radii_in_voxels = []

    if len(radii_in_voxels) > 0:
        min_r = np.min(radii_in_voxels)
        max_r = np.max(radii_in_voxels)
        # Dinamikus sigmák
        dynamic_sigmas = np.linspace(min(min_r, 1.0), max(max_r, 5.0), num=10).tolist()
    else:
        # Ha egyáltalán nincs érték, adjunk egy biztonságos fallback listát!
        dynamic_sigmas = [1.0, 2.0, 3.0,  4.0, 5.0, 6.0]

    z_idx, y_idx, x_idx = np.nonzero(smoothed)

    z_min: int = max(0, z_idx.min())
    z_max: int = min(smoothed.shape[0], z_idx.max() + 1)
        
    y_min: int = max(0, y_idx.min())
    y_max: int = min(smoothed.shape[1], y_idx.max() + 1)
        
    x_min: int = max(0, x_idx.min())
    x_max: int = min(smoothed.shape[2], x_idx.max() + 1)
    
    # A szűréshez minimalizáljuk a feldolgozandó területet egy bounding box-szal
    
    # CT kivágása
    smoothed_cropped = smoothed[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # A fő vonal alsó 20%-át levágjuk, hogy ez ne zavarjon
    cut_len = int(len(main_path) * 0.2) 
    interesting_path = main_path[cut_len:]
    
    # EREDETI MÉRETŰ maszkot készítünk, beleírjuk az 1-eseket
    interesting_mask_full = np.zeros_like(main_path_mask, dtype=np.uint8)
    if len(interesting_path) > 0:
        interesting_mask_full[tuple(interesting_path.T)] = 1
        
    # Majd ezt VÁGJUK KI a kisebb bounding boxra
    interesting_mask_cropped = interesting_mask_full[z_min:z_max, y_min:y_max, x_min:x_max]
            
    tubeness_result = filter.multiscale_tubeness_3d(
            image=smoothed_cropped.astype(np.float32), 
            sigmas=dynamic_sigmas,
            skeleton=interesting_mask_cropped, 
            alpha=0.25, # annyira nem fontos, hogy a cső két kersztmetszete hasonló legyen 
            beta=1.0
        )
    
   
    
    # interesting_path még az eredeti térben
    path_z_cropped = interesting_path[:, 0] - z_min
    path_y_cropped = interesting_path[:, 1] - y_min
    path_x_cropped = interesting_path[:, 2] - x_min

    # 2. Értékek "kimetszése" a 3D térből PONTOSAN AZ ÚTVONAL SORRENDJÉBEN
    scores_on_path = tubeness_result['scores'][path_z_cropped, path_y_cropped, path_x_cropped]
    eigenvalues_on_path = tubeness_result['eigenvalues'][path_z_cropped, path_y_cropped, path_x_cropped, :]
    eigenvectors_on_path = tubeness_result['eigenvectors'][path_z_cropped, path_y_cropped, path_x_cropped, :]
    
    distance_map_cropped = distance_map[z_min:z_max, y_min:y_max, x_min:x_max]
    radii_on_path = distance_map_cropped[path_z_cropped, path_y_cropped, path_x_cropped]

    max_scores_per_z = {}
    for z, score in zip(path_z_cropped, scores_on_path):
        max_scores_per_z[z] = max(max_scores_per_z.get(z, 0), score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(list(max_scores_per_z.keys()), list(max_scores_per_z.values()), marker='o')
    plt.title('Maximális és átlagos csőszerűségi értékek a skeleton mentén Z szerint')
    plt.xlabel('Z index')
    plt.ylabel('Csőszerűségi érték')
    plt.grid()
    plt.savefig(os.path.join(out_dir, "max_tubeness_along_skeleton.png"))
    plt.show()
    
    # 3. DataFrame összeállítása
    import pandas as pd
    
    l1_vec_z = eigenvectors_on_path[:, 0] # Z komponens
    l1_vec_y = eigenvectors_on_path[:, 1] # Y komponens
    l1_vec_x = eigenvectors_on_path[:, 2] # X komponens

    df_features = pd.DataFrame({
        'Z_index': interesting_path[:, 0], # Eredeti Z
        'Y_index': interesting_path[:, 1], # Eredeti Y
        'X_index': interesting_path[:, 2], # Eredeti X
        'Tubeness': scores_on_path,
        'Radius': radii_on_path,
        'L1': eigenvalues_on_path[:, 0],
        'L2': eigenvalues_on_path[:, 1],
        'L3': eigenvalues_on_path[:, 2],
        'Norm_Z': l1_vec_z, # A vágósík normálvektorának Z komponense
        'Norm_Y': l1_vec_y, # A vágósík normálvektorának Y komponense
        'Norm_X': l1_vec_x  # A vágósík normálvektorának X komponense
    })
    
    df_resampled = resample_dataframe(df_features)
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "skeleton_features_resampled.csv")
        df_resampled.to_csv(csv_path, index=False)

    return df_resampled
