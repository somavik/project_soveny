import numpy as np
from typing import Dict, Tuple

def crop_to_roi(ct_array: np.ndarray, labels_dic: Dict[str, np.ndarray], roi_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Megkeresi a Bounding Box koordinátáit a roi_mask alapján, és egyszerre kivágja 
    a CT tömböt, valamint a szótárban levő összes label maszkot.
    """
    z_idx, y_idx, x_idx = np.nonzero(roi_mask)
    if len(z_idx) == 0:
        return np.array([]), {k: np.array([]) for k in labels_dic}
        
    z_min: int = max(0, z_idx.min())
    z_max: int = min(ct_array.shape[0], z_idx.max() + 1)
    
    y_min: int = max(0, y_idx.min())
    y_max: int = min(ct_array.shape[1], y_idx.max() + 1)
    
    x_min: int = max(0, x_idx.min())
    x_max: int = min(ct_array.shape[2], x_idx.max() + 1)
    
    # CT kivágása
    cropped_ct = ct_array[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Labelek egyenkénti kivágása
    cropped_labels_dic = {}
    for name, mask in labels_dic.items():
        cropped_labels_dic[name] = mask[z_min:z_max, y_min:y_max, x_min:x_max]
        
    return cropped_ct, cropped_labels_dic

def get_cropped_array(array_to_crop: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    """
    Megkeresi a Bounding Box koordinátáit a reference_mask alapján, és kivágja a array_to_crop tömböt.
    """
    z_idx, y_idx, x_idx = np.nonzero(reference_mask)
    if len(z_idx) == 0:
        return np.array([])
        
    z_min: int = max(0, z_idx.min())
    z_max: int = min(array_to_crop.shape[0], z_idx.max() + 1)
    
    y_min: int = max(0, y_idx.min())
    y_max: int = min(array_to_crop.shape[1], y_idx.max() + 1)
    
    x_min: int = max(0, x_idx.min())
    x_max: int = min(array_to_crop.shape[2], x_idx.max() + 1)
    
    cropped_array = array_to_crop[z_min:z_max, y_min:y_max, x_min:x_max]
    
    return cropped_array