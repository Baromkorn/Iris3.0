import base64
import os

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == "__main__":
    image_path = input("Enter full path to your image: ").strip('"')

    try:
        b64_string = image_to_base64(image_path)

        # Fixed output directory
        output_dir = r"C:\Users\USER\Desktop\Open-Iris Dev\BMT-20_Base64"
        os.makedirs(output_dir, exist_ok=True)

        # Build output file path with the same name as image but .txt extension
        filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "w") as f:
            f.write(b64_string)

        print(f"✅ Base64 string saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
