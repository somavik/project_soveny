import SimpleITK as sitk
from pathlib import Path
import numpy as np

def save_array_as_image(image_array: np.ndarray, reference_image: sitk.Image, output_path: str):
    """
    Elment egy numpy array-t (pl. tubeness vagy sheetness eredmény) képként, 
    megtartva az eredeti kép (reference_image) térbeli metaadatait.
    """
    out_image = sitk.GetImageFromArray(image_array.astype(np.float32))
    out_image.CopyInformation(reference_image)
    sitk.WriteImage(out_image, output_path)
    print(f"Kép sikeresen elmentve ide: {output_path}")

def derive_output_path(selected_image_path: str) -> str:
    """Generate output output path from selected inputs."""
    selected = Path(selected_image_path)
    file_name = selected.name
    print(f"Selected file: {file_name}")

    if file_name.endswith('.nii.gz'):
        base_name = file_name[:-7]
        extension = '.nii.gz'
    else:
        base_name = selected.stem
        extension = selected.suffix

    if base_name.endswith('_image'):
        base_name = base_name[:-6]

    output_name = f'{base_name}_filtered{extension}'
    return str(selected.with_name(output_name))

def derive_output_dir(selected_image: str, dataset_name: str) -> Path:
    """Generate output directory from selected input."""
    selected = Path(selected_image)
    
    # Kinyerjük a fájlnevét a kiterjesztések nélkül (nii, nii.gz), és eltávolítjuk az "_image" részt ha benne van
    file_name = selected.name.split('.')[0].replace('_image', '')
    
    # Létrehozzuk a kívánt formátumot: output/dataset_name/file_name
    return Path('output') / dataset_name / file_name

_current_reference = None

def set_reference(image: sitk.Image):
    """Beállítja az aktuális páciens CT-jét referenciának."""
    global _current_reference
    _current_reference = image

def save_image(array, output_path):
    """Kimenti a képet, automatikusan a globális referenciát használva."""
    global _current_reference
    if _current_reference is None:
        raise ValueError("HIBA: Nincs beállítva referencia kép a mentéshez!")
        
    new_image = sitk.GetImageFromArray(array)
    new_image.CopyInformation(_current_reference)
    sitk.WriteImage(new_image, output_path)
    return new_image