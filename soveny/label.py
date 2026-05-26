import numpy as np
from typing import Tuple, Dict

def extract_labels(label_array: np.ndarray, config: dict) -> Dict[str, np.ndarray]:
    labels = np.unique(label_array)
    #print(f"Azonosított címkék: {labels}")

    left_ventricle_label: np.ndarray = (label_array == config['labels']['left_ventricle'])
    right_ventricle_label: np.ndarray = (label_array == config['labels']['right_ventricle'])
    aorta_label: np.ndarray = (label_array == config['labels']['aorta'])
    artery_label: np.ndarray = (label_array == config['labels']['artery'])

    return {
        'left_ventricle': left_ventricle_label,
        'right_ventricle': right_ventricle_label,
        'aorta': aorta_label,
        'artery': artery_label,
    }
