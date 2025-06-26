import base64
import sys
from typing import Optional
import os

output_dir = "base64_outputs"  # Change this to whatever folder you want
os.makedirs(output_dir, exist_ok=True)

def image_to_base64(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_images_to_base64.py <left_eye_image_path> [right_eye_image_path]")
        sys.exit(1)

    left_path = sys.argv[1]
    right_path = sys.argv[2] if len(sys.argv) > 2 else None

    left_b64 = image_to_base64(left_path)
    right_b64 = image_to_base64(right_path) if right_path else None

    if left_b64:
        with open(os.path.join(output_dir, "left_iris_base64.txt"), "w") as f:
            f.write(left_b64)
        print(f"Left iris image base64 saved to left_iris_base64.txt")
    else:
        print("Failed to encode left iris image.")

    if right_b64:
        with open(os.path.join(output_dir, "right_iris_base64.txt"), "w") as f:
            f.write(right_b64)
        print(f"Right iris image base64 saved to right_iris_base64.txt")
    elif right_path:
        print("Failed to encode right iris image.")
    else:
        print("No right iris image provided.")

if __name__ == "__main__":
    main()
