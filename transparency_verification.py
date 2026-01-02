import cv2
import sys


def verify_transparency(image_path):
    # Load image including the Alpha Channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Error: Could not read file.")
        return

    # Check for Alpha Channel (4th layer)
    if img.shape[2] == 4:
        # Check the 'Alpha' value of the corner pixel (0,0)
        corner_alpha = img[0, 0, 3]

        print(f"File Type: PNG (Correct)")
        print(f"Channels:  {img.shape[2]} (Red, Green, Blue, Alpha)")

        if corner_alpha == 0:
            print("Status:    ✅ TRANSPARENT (Background is invisible)")
        elif corner_alpha == 255:
            print("Status:    ❌ OPAQUE (Background is solid)")
        else:
            print(f"Status:    ⚠️ SEMI-TRANSPARENT (Alpha: {corner_alpha})")
    else:
        print("Status:    ❌ NO TRANSPARENCY (Only 3 channels found)")


if __name__ == "__main__":
    # Point this to your .png file
    path = input("Enter path to your PNG: ").strip()
    verify_transparency(path)