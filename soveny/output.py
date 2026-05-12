import SimpleITK as sitk
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
