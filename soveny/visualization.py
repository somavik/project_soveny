"""Matplotlib-based visualization helpers wrapped in functions.

Each function accepts optional `show` or `save_path` so they can be disabled when
running headless or during batch processing.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from typing import Optional, Tuple, Union, Dict


def show_bounding_box(image: np.ndarray, mask: np.ndarray, bbox: Tuple[slice, slice, slice], show: bool = True, save_path: Optional[str] = None):
    z_slice = (bbox[0].start + bbox[0].stop) // 2
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image[z_slice], cmap='gray')
    ax[0].set_title('CT slice (center of bbox)')
    ax[1].imshow(mask[z_slice], cmap='gray')
    ax[1].set_title('Mask slice (center of bbox)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def show_tubeness_result(tubeness: np.ndarray, slice_index: Optional[int] = None, show: bool = True, save_path: Optional[str] = None):
    si = slice_index if slice_index is not None else tubeness.shape[0] // 2
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(tubeness[si], cmap='inferno')
    ax.set_title(f'Tubeness slice {si}')
    plt.colorbar(ax.images[0], ax=ax)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_slice_with_labels(ct_array: np.ndarray, labels: Union[np.ndarray, Dict[str, np.ndarray]], axis: str = 'z', slice_idx: Optional[int] = None, show: bool = True, save_path: Optional[str] = None):
    """
    Plots a specific slice of the CT array alongside a colorized label overlay.
    
    Args:
        ct_array: numpy array containing the raw CT.
        labels: numpy array containing the labels, or a dictionary of name -> mask array.
        axis: 'x', 'y' or 'z', indicating the slicing dimension.
        slice_idx: The index of the slice. If None, uses the middle slice.
        show: If True, calls plt.show().
        save_path: Optional path to save the plot.
    """
    axis_map = {'z': 0, 'y': 1, 'x': 2}
    if axis.lower() not in axis_map:
        raise ValueError("axis parameter must be 'x', 'y', or 'z'")
    
    a_idx = axis_map[axis.lower()]
    
    if slice_idx is None:
        slice_idx = ct_array.shape[a_idx] // 2

    # Ha dictionary-t kaptunk, csinálunk belőle egy összevont array-t
    label_array = np.zeros_like(ct_array, dtype=int)
    label_names = []
    
    if isinstance(labels, dict):
        for i, (name, mask) in enumerate(labels.items(), start=1):
            label_array[mask > 0] = i
            label_names.append(name)
    else:
        label_array = labels

    if axis.lower() == 'z':
        ct_slice = ct_array[slice_idx, :, :]
        label_slice = label_array[slice_idx, :, :]
    elif axis.lower() == 'y':
        ct_slice = ct_array[:, slice_idx, :]
        label_slice = label_array[:, slice_idx, :]
    else:  # 'x'
        ct_slice = ct_array[:, :, slice_idx]
        label_slice = label_array[:, :, slice_idx]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # 1. Nyers CT
    ax[0].imshow(ct_slice, cmap='gray')
    ax[0].set_title(f'CT (Axis: {axis.upper()}, Slice: {slice_idx})')
    ax[0].axis('off')
    
    # 2. CT és Overlay
    ax[1].imshow(ct_slice, cmap='gray')
    # A 0-kat maszkoljuk, hogy ezen a részen áttetsző maradjon és átüssön alóla a CT képe
    masked_label = np.ma.masked_where(label_slice == 0, label_slice)
    
    # Diszkrét colormap beállítása, ha több címke van
    max_label = int(np.max(label_slice)) if np.max(label_slice) > 0 else 1
    cmap = plt.get_cmap('tab10', max_label)
    
    ax[1].imshow(masked_label, cmap=cmap, alpha=0.5, interpolation='nearest', vmin=0.5, vmax=max_label+0.5) 
    ax[1].set_title('CT with Label Overlay')
    ax[1].axis('off')

    # Ha dict volt (nevekkel), kirakunk egy legendet
    if isinstance(labels, dict) and label_names:
        patches = [mpatches.Patch(color=cmap(i / max_label), label=name) for i, name in enumerate(label_names)]
        ax[1].legend(handles=patches, loc='upper right', bbox_to_anchor=(1.3, 1))
        # Hogy elférjen a legend a képen:
        plt.subplots_adjust(right=0.85)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

