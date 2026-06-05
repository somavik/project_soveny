import os
from networkx import radius
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from skimage.draw import line_nd
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, distance_transform_edt
import SimpleITK as sitk
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


from soveny import filter
from soveny import output
from soveny import visualization

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import splprep, splev
import networkx as nx

def get_shortest_path_between_farest_points(skeleton: np.ndarray, smooth_factor: float = 10.0) -> np.ndarray:
    """
    Megkeresi a csontváz (skeleton) két legtávolabbi pontját, 
    kiszámolja a valódi fizikai legrövidebb utat (Dijkstra), 
    majd egy B-spline algoritmussal kisimítja azt.
    """
    # 1. Koordináták kinyerése a 3D tömbből (N x 3 mátrix)
    coords = np.argwhere(skeleton)
    if len(coords) < 2:
        return coords

    # 2. Legtávolabbi pontok gyors keresése (C-optimalizált pdist)
    # Sokkal gyorsabb, mint a dupla Python for ciklus.
    dist_matrix = squareform(pdist(coords))
    start_idx, end_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    
    start_point = tuple(coords[start_idx])
    end_point = tuple(coords[end_idx])

    # 3. Szomszédok keresése KDTree-vel (Villámgyors 26-szomszédság)
    # A sqrt(3) ~ 1.732 a maximális távolság egy 3x3x3-as rácson (átló)
    # 1.8 sugár pont lefedi az összes 26-os szomszédot
    tree = KDTree(coords)
    pairs = tree.query_pairs(r=1.8)

    # 4. Gráf építése és Dijkstra algoritmus (Valós súlyozott élekkel)
    G = nx.Graph()
    for coord in coords:
        G.add_node(tuple(coord))

    for i, j in pairs:
        p1 = tuple(coords[i])
        p2 = tuple(coords[j])
        # Fizikai Euklideszi távolság kiszámítása az él súlyához (1.0, 1.41, 1.73)
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        G.add_edge(p1, p2, weight=dist)

    try:
        # networkx Dijkstra implementációja a legrövidebb útra
        path = nx.shortest_path(G, source=start_point, target=end_point, weight='weight')
    except nx.NetworkXNoPath:
        print("Figyelem: A skeleton nem összefüggő! Visszatérés egyenes vonallal.")
        return np.array([start_point, end_point])

    path_array = np.array(path)

    # Biztosítjuk a megfelelő irányt (Z koordináta növekvő/csökkenő)
    if path_array[0][0] > path_array[-1][0]:
        path_array = path_array[::-1]

    # 5. Útvonal simítása (B-spline vasalás)
    # Csak akkor simítunk, ha elég hosszú a vonal (köbös spline-hoz min 4 pont kell)
    if len(path_array) > 4:
        # Szétválasztjuk Z, Y, X tengelyekre
        z, y, x = path_array[:, 0], path_array[:, 1], path_array[:, 2]
        
        try:
            # splprep: B-spline görbe illesztése. 
            # A 'smooth_factor' (s paraméter) szabályozza, mennyire kövesse a nyers pontokat.
            # Minél nagyobb az s, annál egyenesebb/simább a görbe. s=0 esetén minden ponton átmegy.
            tck, u = splprep([z, y, x], s=smooth_factor, k=3)
            
            # Újrageneráljuk a pontokat a görbe mentén (ugyanannyi pontot, mint eredetileg)
            u_new = np.linspace(0, 1, len(path_array))
            z_smooth, y_smooth, x_smooth = splev(u_new, tck)
            
            # Összerakjuk a simított mátrixot
            smoothed_path = np.vstack((z_smooth, y_smooth, x_smooth)).T
            
            # Mivel a Spline lebegőpontos (float) koordinátákat ad, vissza kell kerekítenünk 
            # integer indexekké, hogy a mátrixos indexelés (pl. path_mask[tuple(...)]) működjön.
            path_array = np.round(smoothed_path).astype(int)
            
        except Exception as e:
            print(f"Figyelem: A spline simítás nem sikerült ({e}), nyers útvonalat adunk vissza.")

    return path_array


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

    # Normálvektorok újra-normálása (nlerp)
    if 'Norm_Z' in df_resampled.columns and 'Norm_Y' in df_resampled.columns and 'Norm_X' in df_resampled.columns:
        magnitudes = np.sqrt(df_resampled['Norm_Z']**2 + df_resampled['Norm_Y']**2 + df_resampled['Norm_X']**2)
        # Nullával osztás elkerülése:
        magnitudes[magnitudes == 0] = 1.0 

        df_resampled['Norm_Z'] /= magnitudes
        df_resampled['Norm_Y'] /= magnitudes
        df_resampled['Norm_X'] /= magnitudes

    return df_resampled

def get_smoothed_ventricle(ventricle_label: np.ndarray, output_dir: str = None) -> np.ndarray:
    iterations = 2
    
    smoothed_ventricle = ndimage.binary_dilation(ventricle_label, iterations=iterations)
    smoothed_ventricle = ndimage.binary_erosion(smoothed_ventricle, iterations=iterations)
    
    smoothed_ventricle = ventricle_label
    
    if output_dir:
        output.save_image(smoothed_ventricle.astype(np.uint8), os.path.join(output_dir, "smoothed_ventricle.nii.gz"))
    return smoothed_ventricle

def get_smoothed_tube(tube_label: np.ndarray, output_dir: str = None):
    iterations = 2
    
    smoothed_tube = ndimage.binary_erosion(tube_label, iterations=iterations)
    smoothed_tube = ndimage.binary_dilation(smoothed_tube, iterations=iterations)
    
    smoothed_tube = tube_label
    
    if smoothed_tube.sum() == 0:
        print("Figyelmeztetés: A cső maszk teljesen eltűnt a simítás során! Visszaadom az eredetit.")
        return tube_label.astype(np.uint8)
    
    if output_dir:
        output.save_image(smoothed_tube.astype(np.uint8), os.path.join(output_dir, "smoothed_tube.nii.gz"))
    return smoothed_tube


def get_smoothed_unio(scale: float, smoothed_ventricle: np.ndarray, dist_to_ventricle: np.ndarray, extended_tube: np.ndarray, output_dir: str = None) -> np.ndarray:
    distance = 1 * scale
    
    print(f"Skála: {scale}, Távolság a kamrához és csőhöz: {distance}")
    
    
    tubular_tube = (dist_to_ventricle < distance) & extended_tube
    
    unio = smoothed_ventricle | tubular_tube
    
    smoothed = ndimage.binary_dilation(unio, iterations=6)
    smoothed = ndimage.binary_erosion(smoothed, iterations=6)
    
    if output_dir:
        output.save_image(smoothed.astype(np.uint8), os.path.join(output_dir, "smoothed_unio.nii.gz"))
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
    
    os.makedirs(out_dir, exist_ok=True)
    scale = 5.0

    smoothed_ventricle = get_smoothed_ventricle(ventricle_label, output_dir=out_dir)
    dist_to_ventricle = distance_transform_edt(~smoothed_ventricle)
    
    smoothed_tube = get_smoothed_tube(tube_label, output_dir=out_dir)
    while True:

        smoothed = get_smoothed_unio(scale=scale, smoothed_ventricle=smoothed_ventricle, dist_to_ventricle=dist_to_ventricle, extended_tube=smoothed_tube)
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
            if tube_label[main_path_mask.astype(bool)].sum() > 1:
                dilated_skeleton = ndimage.binary_dilation(main_skeleton, iterations=1)
                output.save_image(dilated_skeleton.astype(np.uint8), os.path.join(out_dir, "dilated_main_skeleton.nii.gz"))
                output.save_image(smoothed.astype(np.uint8), os.path.join(out_dir, "smoothed.nii.gz"))
                dilated_main_path = ndimage.binary_dilation(main_path_mask, iterations=1)
                output.save_image(dilated_main_path.astype(np.uint8), os.path.join(out_dir, "dilated_main_path.nii.gz"))
                break
            else:
                scale += 0.5
        else:
            scale += 1.0
            
    # Region of Interest meghatározása az átmenet kereséséhez, a fő vonal felső 10%-a
    cut_len = int(len(main_path) * 0.85) 
    bottleneck_path = main_path[cut_len:]
    
    # ellenörzésképpen mentjük a fő vonalat és a bottleneck régiót is
    bottleneck_mask = np.zeros_like(main_path_mask, dtype=np.uint8)
    if len(bottleneck_path) > 0:
        bottleneck_mask[tuple(bottleneck_path.T)] = 1
    # dilatáljuk egy kicsit, hogy jobban látszódjon a vizualizációkon
    bottleneck_mask = ndimage.binary_dilation(bottleneck_mask, iterations=1)
    output.save_image(bottleneck_mask.astype(np.uint8), os.path.join(out_dir, "bottleneck_mask.nii.gz"))

    distance_map = distance_transform_edt(smoothed.astype(bool))
    if len(bottleneck_path) > 0:
        radii_in_voxels = distance_map[tuple(bottleneck_path.T)] # át kell alakítani koordinátákká
    else:
        radii_in_voxels = []

    if len(radii_in_voxels) > 0:
        min_r = np.min(radii_in_voxels)
        max_r = np.max(radii_in_voxels)
        mean_r = np.mean(radii_in_voxels)
        median_r = np.median(radii_in_voxels)
        # Dinamikus sigmák
        dynamic_sigmas = np.linspace(max(min_r, 2.0), max(max_r, 7.0), num=5).tolist()
        #dynamic_sigmas = np.linspace(mean_r, median_r, num=10).tolist()
        print(f"Dinamikus sigmák: {dynamic_sigmas}")
    else:
        # Ha egyáltalán nincs érték, adjunk egy biztonságos fallback listát!
        dynamic_sigmas = [2.0, 3.0,  4.0, 5.0, 6.0, 7.0]

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
    #cut_len = int(len(main_path) * 0.2) 
    interesting_path = main_path # proba
    
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
            alpha=0.2, # annyira nem fontos, hogy a cső két kersztmetszete hasonló legyen 
            beta=0.6
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
    
    if out_dir:
        plt.figure(figsize=(10, 6))
        # A X tengely mostantól egyszerűen a pontok sorrendje (0, 1, 2, ... N) lesz
        plt.plot(range(len(scores_on_path)), scores_on_path, marker='o', color='blue')
        
        plt.title('Csőszerűségi értékek az útvonal MENTÉN (nem Z szerint)')
        plt.xlabel('Pálya index (lépésszám a vonalon)')
        plt.ylabel('Csőszerűségi érték')
        plt.grid()
        plt.savefig(os.path.join(out_dir, "tubeness_along_skeleton.png"))
    
    # Előjel-ugrások (Hessian-átok) javítása a normálvektorok térbeli folytonosságához
    # Mivel a sajátvektor-felbontásnál v és -v is érvényes, szomszédos pontoknál előfordulhat 180 fokos ugrás
    for i in range(1, len(eigenvectors_on_path)):
        if np.dot(eigenvectors_on_path[i], eigenvectors_on_path[i-1]) < 0:
            eigenvectors_on_path[i] *= -1
            
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
    
    csv_path = os.path.join(out_dir, "skeleton_features_resampled.csv")
    df_resampled.to_csv(csv_path, index=False)
    
    print(df_resampled)
    
    # --- Új Plot: y tengelyen a Z koordináták, x tengelyen az Y koordináták ---
    plt.figure(figsize=(10, 6))
    
    # Adatok kinyerése (itt a resampled DataFrame-t használjuk, hogy egyezzen a kimenettel)
    y_coords = df_resampled['Y_index']
    z_coords = df_resampled['Z_index']
    tubeness_scores = df_resampled['Tubeness']
    
    # Vonal kirajzolása halványan a pontok mögé, hogy látszódjon a pálya folytonossága
    plt.plot(y_coords, z_coords, color='gray', linestyle='--', alpha=0.5, zorder=1)
    
    # Scatter plot a heatmap szerű megjelenítéshez
    # A cmap='magma' vagy 'turbo' nagyon jól mutat orvosi képek analízisénél
    scatter = plt.scatter(y_coords, z_coords, c=tubeness_scores, cmap='magma', 
                          s=60, edgecolor='black', zorder=2)
    
    # Színmagyarázat (colorbar) hozzáadása
    cbar = plt.colorbar(scatter)
    cbar.set_label('Csőszerűségi érték (Tubeness)')
    
    plt.title('Fővonal Y-Z síkon csőszerűség alapján színezve')
    plt.xlabel('Y koordináta')
    plt.ylabel('Z koordináta')
    plt.grid(True)
    
    # Új fájlnév, hogy ne írja felül az előzőt
    plt.savefig(os.path.join(out_dir, "tubeness_yz_heatmap.png"))
    plt.close() # Memóriaszivárgás elkerülése miatt érdemes lezárni a plotot
    

    return df_resampled

def get_cutting_plane(normal_vector, p_z, p_y, p_x, relevant_labels, ct_array, ventricle_type, tube_type, save_path_3d) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(save_path_3d):
        os.makedirs(save_path_3d)
    
    # --- SÍK GENERÁLÁSA ---
    n_z, n_y, n_x = normal_vector[0], normal_vector[1], normal_vector[2]

    # NINCS SZÜKSÉG VISSZAÁLLÍTÁSRA! A p_z, p_y, p_x már az eredeti térben van.
    print(f"Vágási pont: Z={p_z}, Y={p_y}, X={p_x}, Normál vektor: {normal_vector}")

    relevant_roi = relevant_labels[ventricle_type] | relevant_labels[tube_type]
    shape = relevant_roi.shape
    Z, Y, X = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
        
    plane_equation = n_z * (Z - p_z) + n_y * (Y - p_y) + n_x * (X - p_x)

    # 2.0 vastagságú maszk a vizualizációhoz
    visual_plane_mask = np.abs(plane_equation) <= 2.0

    # Irány korrigálása, hogy a Z tengely (fej felé) vágjon lefelé
    if n_z > 0:
        plane_equation = -plane_equation

    half_space_mask = plane_equation > 0 
    
    # vizualizációs ellenőrzéshez
    print("A vágási sík a cső és a kamra között legenerálva.")
    
    # Középpontnak a vágási pontot használjuk, hogy biztosan lássuk a lényeget
    p_z_int, p_y_int, p_x_int = int(round(p_z)), int(round(p_y)), int(round(p_x))
    
    visualization.plot_3d_slices_with_labels(
        ct_array, 
        {**relevant_labels, 'cutting_plane': visual_plane_mask}, 
        slice_indices=(p_z_int, p_y_int, p_x_int),
        save_path=save_path_3d
    )
    output.save_image(visual_plane_mask.astype(np.uint8), os.path.join(save_path_3d, "cutting_plane_visual.nii.gz"))

    return visual_plane_mask, half_space_mask
