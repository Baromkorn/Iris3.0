import cv2

def save_file(content: bytes, filename: str):
    with open(filename, "wb") as f:
        f.write(content)

def load_image(filename: str):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
