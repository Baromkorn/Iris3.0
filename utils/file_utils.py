import cv2

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
