import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def compute_hessian_at_coords(image: np.ndarray, coords: tuple, sigma: float) -> np.ndarray:
    """
    Kiszámolja a Hesse-mátrixot a megadott koordinátákra egy adott skálán (sigma).
    Kimenete: (Pontok_száma, 3, 3) alakú numpy tömb.
    """
    sz, sy, sx = coords
    num_points = len(sz)

    # A deriváltakat a teljes képen számoljuk a Gauss elmosás miatt
    s2 = sigma**2
    Dzz = ndimage.gaussian_filter(image, sigma, order=[2, 0, 0]) * s2
    Dyy = ndimage.gaussian_filter(image, sigma, order=[0, 2, 0]) * s2
    Dxx = ndimage.gaussian_filter(image, sigma, order=[0, 0, 2]) * s2
    Dzy = ndimage.gaussian_filter(image, sigma, order=[1, 1, 0]) * s2
    Dzx = ndimage.gaussian_filter(image, sigma, order=[1, 0, 1]) * s2
    Dyx = ndimage.gaussian_filter(image, sigma, order=[0, 1, 1]) * s2

    # Hesse-mátrix építése CSAK a koordináták pontjainál
    H_skel = np.zeros((num_points, 3, 3), dtype=np.float32)
    H_skel[:, 0, 0] = Dzz[sz, sy, sx]
    H_skel[:, 0, 1] = Dzy[sz, sy, sx]
    H_skel[:, 0, 2] = Dzx[sz, sy, sx]
    H_skel[:, 1, 0] = Dzy[sz, sy, sx]
    H_skel[:, 1, 1] = Dyy[sz, sy, sx]
    H_skel[:, 1, 2] = Dyx[sz, sy, sx]
    H_skel[:, 2, 0] = Dzx[sz, sy, sx]
    H_skel[:, 2, 1] = Dyx[sz, sy, sx]
    H_skel[:, 2, 2] = Dxx[sz, sy, sx]

    return H_skel

def compute_hessian_full(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Kiszámolja a teljes Hesse-mátrixot a kép minden pontjára egy adott skálán (sigma).
    Kimenete: (Z, Y, X, 3, 3) alakú numpy tömb.
    """
    s2 = sigma**2
    Dzz = ndimage.gaussian_filter(image, sigma, order=[2, 0, 0]) * s2
    Dyy = ndimage.gaussian_filter(image, sigma, order=[0, 2, 0]) * s2
    Dxx = ndimage.gaussian_filter(image, sigma, order=[0, 0, 2]) * s2
    Dzy = ndimage.gaussian_filter(image, sigma, order=[1, 1, 0]) * s2
    Dzx = ndimage.gaussian_filter(image, sigma, order=[1, 0, 1]) * s2
    Dyx = ndimage.gaussian_filter(image, sigma, order=[0, 1, 1]) * s2

    H = np.zeros(image.shape + (3, 3), dtype=np.float32)
    H[..., 0, 0] = Dzz; H[..., 0, 1] = Dzy; H[..., 0, 2] = Dzx
    H[..., 1, 0] = Dzy; H[..., 1, 1] = Dyy; H[..., 1, 2] = Dyx
    H[..., 2, 0] = Dzx; H[..., 2, 1] = Dyx; H[..., 2, 2] = Dxx
    return H

def calculate_c_in_skeleton(binary: np.ndarray, skeleton: np.ndarray, sigmas: list) -> float:
    """
    Kiszámolja a c paramétert a csőszerűséghez képlethez a kép intenzitásának statisztikái alapján.
    """
    global_max_S = 0
    skeleton_coords = np.nonzero(skeleton)
    for s in sigmas:
        #print("Mátrix építés és sajátérték-számítás...")
        
        H_skel = compute_hessian_at_coords(binary, skeleton_coords, s)
        eigvals = np.linalg.eigvalsh(H_skel)

        # S = H abs értékének kiszámítása
        S = np.sqrt(np.sum(eigvals**2, axis=-1))
        
        # A zaj kiszűrése és a max/percentilis keresése
        valid_S = S[S > 1e-6]
        if len(valid_S) > 0:
            current_max = np.percentile(valid_S, 99)
        else:
            current_max = 0.0
        
        #print(f"Skála: {s}, 99%-os percentilis S: {current_max}")
            
        if current_max > global_max_S:
            global_max_S = current_max

    c = global_max_S / 2
    c = max(c, 1e-6)
    return c

from scipy.ndimage import distance_transform_edt

def calculate_c(image: np.ndarray, sigmas: list) -> float:
    """
    Kiszámolja a c paramétert a Sheetness képlethez a kép intenzitásának statisztikái alapján.
    """
    global_max_S = 0
    for s in sigmas:
        #print("Mátrix építés és sajátérték-számítás full image-re...")
        
        H = compute_hessian_full(image, s)
        eigvals = np.linalg.eigvalsh(H)

        # S = H abs értékének kiszámítása
        S = np.sqrt(np.sum(eigvals**2, axis=-1))
        
        # A zaj kiszűrése és a max/percentilis keresése
        valid_S = S[S > 1e-6]
        if len(valid_S) > 0:
            current_max = np.percentile(valid_S, 99)
        else:
            current_max = 0.0
        
        #print(f"Skála: {s}, 99%-os percentilis S: {current_max}")
            
        if current_max > global_max_S:
            global_max_S = current_max

    c = global_max_S / 2.0
    c = max(c, 1e-6)
    return c

def evaluate_tubeness_on_skeleton(l1: np.ndarray, l2: np.ndarray, l3: np.ndarray, alpha: float, beta: float, c: float) -> np.ndarray:
    eps = 1e-10
        
    abs_l1, abs_l2, abs_l3 = np.abs(l1), np.abs(l2), np.abs(l3)

    R_plate = abs_l2 / (abs_l3 + eps)
    R_blob = abs_l1 / (np.sqrt(abs_l2 * abs_l3) + eps)
    S = np.sqrt(l1**2 + l2**2 + l3**2)

    scores = (1 - np.exp(-(R_plate**2) / (2 * alpha**2))) * \
             np.exp(-(R_blob**2) / (2 * beta**2)) * \
             (1 - np.exp(-(S**2) / (2 * c**2)))
    #scores[(l3 > -0.2) | (l2 > -0.15)] = 0 
            
    return np.nan_to_num(scores)

def calculate_sheetness_3d(l1: np.ndarray, l2: np.ndarray, l3: np.ndarray, alpha: float, beta: float, c: float) -> np.ndarray:
    eps = 1e-10
    abs_l1, abs_l2, abs_l3 = np.abs(l1), np.abs(l2), np.abs(l3)

    R_plate = abs_l2 / (abs_l3 + eps)
    R_blob = abs_l1 / (np.sqrt(abs_l2 * abs_l3) + eps)
    S = np.sqrt(l1**2 + l2**2 + l3**2)

    scores = np.exp(-(R_plate**2) / (2 * alpha**2)) * \
             np.exp(-(R_blob**2) / (2 * beta**2)) * \
             (1 - np.exp(-(S**2) / (2 * c**2)))
    scores[l3 < 0] = 0
            
    return np.nan_to_num(scores)

def multiscale_sheetness_3d(image: np.ndarray, sigmas: list, alpha=0.5, beta=0.5, c=None) -> np.ndarray:
    """
    Többskálás szűrés (sheetness) a teljes képen.
    """
    if c is None:
        c = calculate_c(image, sigmas)
        #print(f"Használt c érték a sheetness képlethez: {c}")

    max_scores = np.zeros_like(image, dtype=np.float32)

    for s in sigmas:
        #print(f" -> Számítás sigma = {s} skálán (sheetness)...")
        
        H = compute_hessian_full(image, s)

        # Sajátértékek kiszámítása (a sajátvektorokat nem kérjük le)
        eigvals = np.linalg.eigvalsh(H)

        abs_eigvals = np.abs(eigvals)
        sort_indices = np.argsort(abs_eigvals, axis=-1)

        l1 = np.take_along_axis(eigvals, sort_indices[..., 0:1], axis=-1)[..., 0]
        l2 = np.take_along_axis(eigvals, sort_indices[..., 1:2], axis=-1)[..., 0]
        l3 = np.take_along_axis(eigvals, sort_indices[..., 2:3], axis=-1)[..., 0]

        scores = calculate_sheetness_3d(l1, l2, l3, alpha, beta, c)

        # Frissítjük a maximumokat
        better_mask = scores > max_scores
        max_scores[better_mask] = scores[better_mask]

    #print("\nSzűrés kész! Maximális válaszok kigyűjtve.")
    return max_scores

def multiscale_tubeness_3d(image: np.ndarray, skeleton: np.ndarray, sigmas: list, alpha=0.5, beta=0.1, c=None) -> dict:
    """
    Többskálás szűrés (tubeness) a skeleton pontjain.
    """
    if c is None:
        c = calculate_c_in_skeleton(image, skeleton, sigmas)
        #print(f"Használt c érték a tubeness képlethez: {c}")

    max_scores = np.zeros_like(image, dtype=np.float32)
    best_eigenvectors = np.zeros(image.shape + (3,), dtype=np.float32)
    best_eigenvalues = np.zeros(image.shape + (3,), dtype=np.float32)

    coords = np.nonzero(skeleton)
    sz, sy, sx = coords
    num_points = len(sz)
    #print(f"Összesen {num_points} darab skeleton pontot vizsgálunk.")

    for s in sigmas:
        #print(f" -> Számítás sigma = {s} skálán (tubeness)...")
        
        H = compute_hessian_at_coords(image, coords, s)

        # Sajátértékek és sajátvektorok kiszámítása
        eigvals, eigvecs = np.linalg.eigh(H)

        abs_eigvals = np.abs(eigvals)
        sort_indices = np.argsort(abs_eigvals, axis=-1)

        l1 = np.take_along_axis(eigvals, sort_indices[..., 0:1], axis=-1)[..., 0]
        l2 = np.take_along_axis(eigvals, sort_indices[..., 1:2], axis=-1)[..., 0]
        l3 = np.take_along_axis(eigvals, sort_indices[..., 2:3], axis=-1)[..., 0]

        # L1 sajátvektor (a vágósík normálvektora)
        u1_indices = sort_indices[..., 0]
        # U1 vektor kiválasztása, id_repeated kell a megfelelő dimenziószám miatt
        idx_repeated = np.repeat(np.expand_dims(u1_indices, axis=(-2, -1)), 3, axis=-2)
        u1_current = np.take_along_axis(eigvecs, idx_repeated, axis=-1)[..., 0]

        scores = evaluate_tubeness_on_skeleton(l1, l2, l3, alpha, beta, c)

        # Visszamappelés a teljes 3D képre
        scores_full = np.zeros_like(image, dtype=np.float32)
        scores_full[sz, sy, sx] = scores
        scores = scores_full

        u1_full = np.zeros(image.shape + (3,), dtype=np.float32)
        u1_full[sz, sy, sx, :] = u1_current
        u1_current = u1_full

        l_full = np.zeros(image.shape + (3,), dtype=np.float32)
        l_full[sz, sy, sx, :] = np.stack([l1, l2, l3], axis=-1)

        # Frissítjük a maximumokat
        better_mask = scores > max_scores
        max_scores[better_mask] = scores[better_mask]
        best_eigenvectors[better_mask] = u1_current[better_mask]
        best_eigenvalues[better_mask] = l_full[better_mask]

    #print("\nSzűrés kész! Maximális válaszok kigyűjtve.")
    return {
        'scores': max_scores,
        'eigenvectors': best_eigenvectors,
        'eigenvalues': best_eigenvalues
    }
