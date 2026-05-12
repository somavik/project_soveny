"""Orchestrator script: ties modules together into a simple pipeline.

Usage (example):
    python -m soveny.main --auto
"""
import argparse
import os
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

def main(dataset_name: str = 'ImageCHD_dataset'):
    parser = argparse.ArgumentParser(description='Run the Soveny pipeline on a selected image and label.')
    parser.add_argument('--dataset', type=str, default=dataset_name, help=f'Name of the dataset to load (default: {dataset_name})')
    args = parser.parse_args()

    dataset_dir, cfg = config.load_config(args.dataset)
    image_path, label_path = input.get_input_paths(dataset_dir)
    ct_image, ct_array, label_image, label_array = input.load_ct_and_label(image_path, label_path)
    relevant_labels_dic = label.extract_labels(label_array, cfg)
    
    roi_mask = relevant_labels_dic['left_ventricle'] | relevant_labels_dic['right_ventricle']
    
    cropped_ct_array, cropped_relevant_labels_dic = ventircles_BB.crop_to_roi(
        ct_array, 
        relevant_labels_dic, 
        roi_mask
    )
    
    visualization.plot_slice_with_labels(cropped_ct_array, cropped_relevant_labels_dic, axis='z', save_path=os.path.join("plots", 'ventricles_overlay.png'))

    septum_mask = septum_BB.get_dilated_intersection(
        cropped_relevant_labels_dic['left_ventricle'],
        cropped_relevant_labels_dic['right_ventricle'],
        iterations=55
    )
    
    visualization.plot_slice_with_labels(cropped_ct_array, {'septum': septum_mask}, axis='z', save_path=os.path.join("plots", 'septum_overlay.png'))

    visual_plane_aorta, cut_mask_aorta = tube_BB.get_cutting_plane(
        ventricle_label=relevant_labels_dic['left_ventricle'],
        tube_label=relevant_labels_dic['aorta'],
        ct_array=ct_array,
        ct_image=ct_image,
        out_dir="output",
        tube_type='aorta',
        ventricle_type='left_ventricle'
    )
    
    visual_plane_artery, cut_mask_artery = tube_BB.get_cutting_plane(
       ventricle_label=relevant_labels_dic['right_ventricle'],
        tube_label=relevant_labels_dic['artery'],
        ct_array=ct_array,
        ct_image=ct_image,
        out_dir="output",
        tube_type='artery',
        ventricle_type='right_ventricle'
    )
    
    cropped_cut_mask_aorta = ventircles_BB.get_cropped_array(cut_mask_aorta, roi_mask)
    cropped_cut_mask_artery = ventircles_BB.get_cropped_array(cut_mask_artery, roi_mask)
    
    visualization.plot_slice_with_labels(ct_array, {'aorta_cut': cut_mask_aorta, 'artery_cut': cut_mask_artery}, axis='z', save_path=os.path.join("plots", 'cut_masks_overlay.png'))
    
    # Ablakozás (windowing) a CT értékeken, ami a notebook-ban is javítja a sheetness eredményét
    cropped_ct_windowed = np.clip(cropped_ct_array, 1000, 1600)
    
    # Filter lefuttatása a windowolt CT-n dinamikusan számolt 'c' konstanssal 
    max_scores = filter.multiscale_sheetness_3d(cropped_ct_windowed, sigmas=[1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], alpha=0.5, beta=0.5)
    
    # A levágott végleges tiszta bal kamra (az aortalefolyás nélkül)
    combined_mask = np.logical_and(np.logical_and(cropped_cut_mask_aorta, cropped_cut_mask_artery), septum_mask)
    cropped_sheetness = np.where(combined_mask, max_scores, 0)
    
    visualization.plot_slice_with_labels(cropped_sheetness, cropped_relevant_labels_dic, axis='z', save_path=os.path.join("plots", 'final_sheetness_overlay.png'))
    
    # Visszatesszük a kivágott sheetness eredményt a teljes CT méretű tömbbe
    sheetness_full = np.zeros_like(ct_array, dtype=np.float32)
    z_idx, y_idx, x_idx = np.nonzero(roi_mask)
    if len(z_idx) > 0:
        z_min, z_max = max(0, z_idx.min()), min(ct_array.shape[0], z_idx.max() + 1)
        y_min, y_max = max(0, y_idx.min()), min(ct_array.shape[1], y_idx.max() + 1)
        x_min, x_max = max(0, x_idx.min()), min(ct_array.shape[2], x_idx.max() + 1)
        sheetness_full[z_min:z_max, y_min:y_max, x_min:x_max] = cropped_sheetness
    
    output.save_array_as_image(sheetness_full, ct_image, os.path.join(dataset_dir, 'final_sheetness.nii.gz'))
    
if __name__ == '__main__':
    main()
