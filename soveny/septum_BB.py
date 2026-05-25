import numpy as np
import scipy.ndimage as ndimage

def get_dilated_intersection(mask1: np.ndarray, mask2: np.ndarray, lin_heart_size: int = 200000) -> np.ndarray:
    """
    Kiszélesíti (dilatálja) a két bemeneti maszkot a megadott iterációszámmal,
    majd visszaadja a két dilatált maszk metszetét (AND).
    """
    
    # 3. Az iterációk száma legyen a szívméret egy kis százaléka (pl. 12%)
    # Egy felnőtt szíve kb. 300-400 ezer mm3 -> linear_size ~ 70 mm -> 70 * 0.12 = ~8-9 iteráció (mm)
    iterations = int(np.ceil(lin_heart_size * 0.4))  # 12% a linear size-ból, kerekítve felfelé
    
    # Minimum és maximum korlát (Biztonsági háló)
    iterations = np.clip(iterations, a_min=5, a_max=30)
    
    print(f"Dilatáció iterációk száma: {iterations}")
    mask1_dilated = ndimage.binary_dilation(mask1, iterations=iterations)
    mask2_dilated = ndimage.binary_dilation(mask2, iterations=iterations)      

    septum_bounding_region = np.logical_and(mask1_dilated, mask2_dilated)
    return septum_bounding_region


def get_septum_by_distance(mask1: np.ndarray, mask2: np.ndarray, max_distance_mm: int = 11) -> np.ndarray:
    """
    Távolságtérképek alapján keresi meg a sövényt. 
    max_distance_mm: Milyen messze lehet egy pixel a kamra falától, hogy még septumként kezeljük?
    """
    # Kiszámolja minden pixel távolságát a mask1-től és mask2-től
    # A ~ (NOT) operátor inverzeli a maszkot, így azokon kívüli távolságot mérjük
    dist_to_1 = ndimage.distance_transform_edt(~mask1)
    dist_to_2 = ndimage.distance_transform_edt(~mask2)
    
    # A septum az a régió, ahol mindkét kamrától legfeljebb 'max_distance_mm' távolságra vagyunk
    # Mivel izotróp a rácsunk (1x1x1 mm), a távolság közvetlenül millimétert jelent!
    septum_region = (dist_to_1 <= max_distance_mm) & (dist_to_2 <= max_distance_mm)
    
    return septum_region