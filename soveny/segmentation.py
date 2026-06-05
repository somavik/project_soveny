import os

import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import torch

models = [
    {
        "name": "605_STSTplusLung_large_full",
        "path": "Dataset605_STSTplusLung_large_full\\nnUNetTrainer__nnUNetPlans__3d_fullres",
        "fold": "all",
        "weights": "checkpoint_final.pth",
    },
]

ct_root_dir = r"tmp_nnunet_in"
res_root_dir = r"tmp_nnunet_out"


def predict_ct(ct: np.ndarray, props, model_folder, folds, weights_file):
    predictor = nnUNetPredictor(
        # tile_step_size=step_size,
        use_gaussian=False,
        use_mirroring=False,
        perform_everything_on_device=False,  # for nnunetv2>=2.2.2
        device=torch.device('cpu'),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True,
    )

    # img = [ct_npy.astype(np.float32)]
    # prop = {'spacing': list(pixel_spacing)}
    img = ct

    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name=weights_file,
    )
    pred, probs = predictor.predict_single_npy_array(img, props, None, None, True)
    return pred, probs


if __name__ == '__main__':
    root_folder = ct_root_dir
    output_folder = res_root_dir

    # Iterate over the root folder
    for root, dirs, files in sorted(os.walk(root_folder, followlinks=True)):
    # Skip .seg.nrrd files
        for file in files:
            if not file.endswith('.nii.gz'): continue
            ct_input_path = os.path.join(root, file)
            img, props = SimpleITKIO().read_images([ct_input_path])

            # Get the top folder name
            relative_path = os.path.relpath(root, root_folder)
            subdir = relative_path.split(os.sep)[0]

            # Create the corresponding processed folder if it doesn't exist
            processed_subfolder = os.path.join(output_folder, subdir)

            for model in models:
                # Define the output path (same filename as input)
                output_path = os.path.join(processed_subfolder, model['name'], file.replace('.nii.gz', '.seg.nrrd'))
                if os.path.exists(output_path):
                	continue
                print(f'{ct_input_path} --> {output_path}')

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                pred, probs = predict_ct(img, props, model['path'], model['fold'], model['weights'])

                SimpleITKIO().write_seg(pred, output_path, props)

                print(f"Processed file saved to: {output_path}")
                    