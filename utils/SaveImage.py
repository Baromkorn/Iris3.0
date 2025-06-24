import os
from datetime import datetime
import cv2
from fastapi import UploadFile
import numpy as np

def save_iris_image(img: np.ndarray, cid: str, eye_side: str, unique_id: str) -> str:
    """
    Save the uploaded iris image as a BMP file into structured folders.

    Args:
        file (UploadFile): The uploaded iris image.
        cid (str): Citizen ID.
        eye_side (str): 'left' or 'right'.

    Returns:
        str: Full path to the saved file.
    """
    now = datetime.now()
    folder_path = os.path.join(
        os.getcwd(), "iris_image", now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"), cid
    )
    os.makedirs(folder_path, exist_ok=True)

    # Read and decode image

    if img is None:
        raise ValueError("Failed to decode image.")
    

    unique_name = f"{unique_id}_{eye_side.upper()}.bmp"
    save_path = os.path.join(folder_path, unique_name)
    cv2.imwrite(save_path, img)

    return save_path