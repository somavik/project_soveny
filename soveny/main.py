"""Orchestrator script: ties modules together into a simple pipeline.

Usage (example):
    python -m soveny.main --auto
"""
import argparse
from ast import arg
import os
from sympy import arg
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
    dataset_dir = os.path.join(dataset_dir, "preprocessed")  # Resampled könyvtár használata most az ImageCHD_dataset-ben, ahol már izotrópra van resample-elve a CT és a label is.
    image_path, label_path = input.get_input_paths(dataset_dir)
    ct_image, ct_array, label_image, label_array = input.load_ct_and_label(image_path, label_path)
    
    output_dir = output.derive_output_dir(image_path, args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    relevant_labels_dic = label.extract_labels(label_array, cfg)
    
    roi_mask = relevant_labels_dic['left_ventricle'] | relevant_labels_dic['right_ventricle']
    
    cropped_ct_array, cropped_relevant_labels_dic = ventircles_BB.crop_to_roi(
        ct_array, 
        relevant_labels_dic, 
        roi_mask
    )
    
    visualization.plot_slice_with_labels(cropped_ct_array, cropped_relevant_labels_dic, axis='z', save_path=os.path.join(output_dir, 'ventricles_overlay.png'))
    
    septum_mask = septum_BB.get_septum_by_distance(
        cropped_relevant_labels_dic['left_ventricle'],
        cropped_relevant_labels_dic['right_ventricle'],
        max_distance_mm=12
    )
    
    visualization.plot_slice_with_labels(cropped_ct_array, {'septum': septum_mask,}, axis='z', save_path=os.path.join(output_dir, 'septum_overlay.png'))

    aorta_df = tube_BB.get_cutting_features(
        ventricle_label=relevant_labels_dic['left_ventricle'],
        tube_label=relevant_labels_dic['aorta'],
        ct_array=ct_array,
        ct_image=ct_image,
        out_dir=os.path.join(output_dir, "aorta_cutting_plane"),
        tube_type='aorta',
        ventricle_type='left_ventricle'
    )
    
    artery_df = tube_BB.get_cutting_features(
       ventricle_label=relevant_labels_dic['right_ventricle'],
        tube_label=relevant_labels_dic['artery'],
        ct_array=ct_array,
        ct_image=ct_image,
        out_dir=os.path.join(output_dir, "artery_cutting_plane"),
        tube_type='artery',
        ventricle_type='right_ventricle'
    )

    print("\nAorta cutting plane features:")
    print(aorta_df)
    print("\nArtery cutting plane features:")
    print(artery_df)
    
    
if __name__ == '__main__':
    main()
