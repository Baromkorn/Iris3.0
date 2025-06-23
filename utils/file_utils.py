import cv2
import numpy as np
from iris import IrisTemplate
def save_file(content: bytes, filename: str):
    with open(filename, "wb") as f:
        f.write(content)

def load_image(filename: str):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

def convert_bool_list_to_bytes(bool_list):
    if len(bool_list) % 8 != 0:
        raise ValueError("The length of a boolean list must be a multiple of 8")

    byte_array = bytearray(len(bool_list) // 8)
    for i, bit in enumerate(bool_list):
        if bit == 1:
            index = i // 8
            shift = i % 8
            byte_array[index] |= (1 << shift)
    return bytes(byte_array)
def convert_bytes_to_bool_list(byte_data):
    bool_list = []
    for byte in byte_data:
        for i in range(8):
            bit = (byte >> i) & 1
            bool_list.append(bit)
    bool_list = np.array(bool_list).reshape(2,16, 256, 2).astype(bool)
    return bool_list
def convert_bytes_to_template(byte_data):
    bool_list = []
    for byte in byte_data:
        for i in range(8):
            bit = (byte >> i) & 1
            bool_list.append(bit)
    bool_list = list(map(bool,bool_list))
    split_vectors = np.array(bool_list).reshape(4, 16, 256, 2).astype(bool)
    iris_codes = [split_vectors[0], split_vectors[1]]
    mask_codes = [split_vectors[2], split_vectors[3]]
    iris_template = IrisTemplate(
        iris_codes=iris_codes,
        mask_codes=mask_codes,
        iris_code_version="v0.1"  # or whatever version you were using
    )
    return iris_template

