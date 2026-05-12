import numpy as np
import scipy.ndimage as ndimage

def get_dilated_intersection(mask1: np.ndarray, mask2: np.ndarray, iterations: int = 40) -> np.ndarray:
    """
    Kiszélesíti (dilatálja) a két bemeneti maszkot a megadott iterációszámmal,
    majd visszaadja a két dilatált maszk metszetét (AND).
    """
    mask1_dilated = ndimage.binary_dilation(mask1, iterations=iterations)
    mask2_dilated = ndimage.binary_dilation(mask2, iterations=iterations)      

    septum_bounding_region = np.logical_and(mask1_dilated, mask2_dilated)
    return septum_bounding_region


