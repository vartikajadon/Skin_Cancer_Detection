import os
import shutil
import cv2
import numpy as np
from pathlib import Path

def main():
    out_dir = Path("test_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating validation dataset in 'test_validation/'...")

    # 1. Copied Positive Skin Lesion Image
    # Look for ISIC_0024306.jpg in data directory
    pos_source = Path("data/HAM10000_images_part_1/ISIC_0024306.jpg")
    pos_dest = out_dir / "lesion_ISIC_0024306.jpg"
    low_conf_dest = out_dir / "lesion_low_conf.jpg"
    
    if pos_source.exists():
        shutil.copy(str(pos_source), str(pos_dest))
        shutil.copy(str(pos_source), str(low_conf_dest))
        print(f"Copied real lesion image to: {pos_dest} and {low_conf_dest}")
    else:
        # Fallback drawing of a simulated skin lesion in case HAM10000 images are missing
        img = np.ones((224, 224, 3), dtype=np.uint8) * 200 # Skin-like background (beige)
        img[:, :, 0] = 180  # B
        img[:, :, 1] = 190  # G
        img[:, :, 2] = 230  # R
        # Draw central dark lesion blob
        cv2.circle(img, (112, 112), 40, (50, 60, 80), -1)
        cv2.imwrite(str(pos_dest), img)
        cv2.imwrite(str(low_conf_dest), img)
        print(f"Generated simulated lesion image to: {pos_dest} and {low_conf_dest}")

    # Helper function to generate a standard OOD image with geometric shapes and text label
    def make_ood_image(filename, label, color):
        img = np.ones((300, 300, 3), dtype=np.uint8) * 50 # Dark background
        # Draw some arbitrary shapes
        cv2.rectangle(img, (50, 50), (250, 250), color, -1)
        cv2.circle(img, (150, 150), 60, (255 - color[0], 255 - color[1], 255 - color[2]), 10)
        # Put text
        cv2.putText(img, label, (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        path = out_dir / filename
        cv2.imwrite(str(path), img)
        print(f"Generated OOD image: {path}")

    # 2. Cat Image
    make_ood_image("cat_1.jpg", "CAT (OOD)", (100, 255, 100))
    # 3. Dog Image
    make_ood_image("dog_1.jpg", "DOG (OOD)", (255, 100, 100))
    # 4. Car Image
    make_ood_image("car_1.jpg", "CAR (OOD)", (100, 100, 255))
    # 5. Building Image
    make_ood_image("building_1.jpg", "BUILDING (OOD)", (200, 50, 200))
    # 6. Human Face Image
    make_ood_image("face_1.jpg", "FACE (OOD)", (50, 200, 200))
    # 7. Food Image
    make_ood_image("food_1.jpg", "FOOD (OOD)", (200, 200, 50))
    # 8. Landscape Image
    make_ood_image("landscape_1.jpg", "LANDSCAPE (OOD)", (255, 150, 50))

    # 9. Empty flat-color image (should trigger quality validation fail)
    empty_img = np.ones((224, 224, 3), dtype=np.uint8) * 128 # Solid gray
    empty_path = out_dir / "empty_flat.jpg"
    cv2.imwrite(str(empty_path), empty_img)
    print(f"Generated empty image: {empty_path}")

    # 10. Blurry image (very strong blur applied to a lesion image)
    blurry_img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    blurry_img[:, :, 0] = 180; blurry_img[:, :, 1] = 190; blurry_img[:, :, 2] = 230
    cv2.circle(blurry_img, (112, 112), 40, (50, 60, 80), -1)
    blurry_img = cv2.GaussianBlur(blurry_img, (51, 51), 0) # Massive blur
    blurry_path = out_dir / "blurry_lesion.jpg"
    cv2.imwrite(str(blurry_path), blurry_img)
    print(f"Generated blurry image: {blurry_path}")

    # 11. Low-resolution image (under 64px)
    low_res_img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    low_res_path = out_dir / "low_res_lesion.jpg"
    cv2.imwrite(str(low_res_path), low_res_img)
    print(f"Generated low-resolution image: {low_res_path}")

    # 12. Extremely dark image
    # Start with mean = 10, add small noise with std_dev = 4.0 to bypass flat empty check
    dark_img = np.random.normal(10, 4.0, (224, 224, 3)).astype(np.uint8)
    dark_img = np.clip(dark_img, 0, 19) # Keep values strictly < 20
    dark_path = out_dir / "dark_lesion.jpg"
    cv2.imwrite(str(dark_path), dark_img)
    print(f"Generated dark image: {dark_path}")

    # 13. Extremely bright image
    # Start with mean = 252.8, draw a single-pixel line to bypass flat and blurry checks
    bright_img = np.ones((224, 224, 3), dtype=np.uint8) * 254
    cv2.line(bright_img, (0, 0), (224, 224), (0, 0, 0), 1)
    bright_path = out_dir / "bright_lesion.jpg"
    cv2.imwrite(str(bright_path), bright_img)
    print(f"Generated bright image: {bright_path}")

    print("Validation dataset generation complete.")

if __name__ == "__main__":
    main()
