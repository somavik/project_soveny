"""I/O utilities: file dialogs and SimpleITK wrappers."""
import os
import tkinter as tk
from pathlib import Path
from tkinter.filedialog import askopenfilename
from typing import Tuple

import SimpleITK as sitk
import numpy as np

def get_input_paths(dataset_dir: str) -> Tuple[str, str]:
    """Select image and label paths via a UI dialog box."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    image_path = askopenfilename(
        title='Select CT image file',
        initialdir=dataset_dir,
        filetypes=[('NIfTI files', '*.nii *.nii.gz'), ('All files', '*.*')]
    )
    if not image_path:
        raise FileNotFoundError('No CT image file selected')

    label_path = askopenfilename(
        title=f'Select label file for {os.path.basename(image_path)}',
        initialdir=dataset_dir,
        filetypes=[('NIfTI files', '*.nii *.nii.gz *.nrrd'), ('All files', '*.*')]
    )
    if not label_path:
        raise FileNotFoundError('No label file selected')

    root.destroy()
    return image_path, label_path

def load_ct_and_label(image_path: str, label_path: str) -> Tuple[sitk.Image, np.ndarray, sitk.Image, np.ndarray]:
    """Read the CT image and label data as SimpleITK objects and numpy arrays."""
    ct_image: sitk.Image = sitk.ReadImage(image_path)
    ct_array: np.ndarray = sitk.GetArrayFromImage(ct_image)
    
    #print(f"Eredeti CT spacing (X, Y, Z): {ct_image.GetSpacing()}")

    label_image: sitk.Image = sitk.ReadImage(label_path)
    label_array: np.ndarray = sitk.GetArrayFromImage(label_image)

    #print(f"Image neve: {os.path.basename(image_path)}")
    #print(f"Label neve: {os.path.basename(label_path)}")

    return ct_image, ct_array, label_image, label_array


