import os
import csv
import random
from pathlib import Path
import numpy as np
import cv2

def setup_demo_data(base_dir: Path, num_records: int = 150):
    """
    Generates a realistic mock HAM10000 dataset structure with metadata and images
    for testing the EDA pipeline.
    """
    print(f"Setting up demo data in: {base_dir.resolve()}")
    
    # Define directories
    data_dir = base_dir / "data"
    part1_dir = data_dir / "HAM10000_images_part_1"
    part2_dir = data_dir / "HAM10000_images_part_2"
    
    part1_dir.mkdir(parents=True, exist_ok=True)
    part2_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata fields
    # dx: 'akiec', 'bcc', 'bkl', 'df', 'nv', 'mel', 'vasc'
    dx_classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    dx_weights = [0.65, 0.11, 0.10, 0.05, 0.04, 0.03, 0.02] # Realistic class imbalance
    
    dx_types = ['histo', 'consensus', 'confocal', 'follow_up']
    localizations = ['back', 'lower extremity', 'trunk', 'upper extremity', 'abdomen', 
                     'face', 'chest', 'foot', 'unknown', 'neck', 'scalp', 'hand', 'ear', 'genital']
    genders = ['male', 'female', 'unknown']
    gender_weights = [0.52, 0.46, 0.02]
    
    csv_path = data_dir / "HAM10000_metadata.csv"
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization'])
        
        for i in range(num_records):
            # Generate IDs
            lesion_id = f"HAM_00{random.randint(1000, 9999)}"
            image_id = f"ISIC_00{24306 + i}"
            
            # Select values
            dx = random.choices(dx_classes, weights=dx_weights, k=1)[0]
            dx_type = random.choice(dx_types)
            
            # Age: introduce some missing values (None/empty) to verify EDA handling
            age = "" if random.random() < 0.05 else float(random.randint(5, 95))
            
            sex = random.choices(genders, weights=gender_weights, k=1)[0]
            localization = random.choice(localizations)
            
            writer.writerow([lesion_id, image_id, dx, dx_type, age, sex, localization])
            
            # Create synthetic skin lesion image
            # Create a 64x64 or 128x128 image with a colored background representing skin
            # and an irregular spot representing a mole
            width = random.choice([64, 128])
            height = width
            
            # Skin background (warm brownish-peach)
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :] = [random.randint(140, 180), random.randint(180, 220), random.randint(220, 255)] # BGR format
            
            # Draw mole
            center_x = width // 2 + random.randint(-5, 5)
            center_y = height // 2 + random.randint(-5, 5)
            radius = random.randint(width // 10, width // 4)
            
            # Generate different color moles depending on dx
            if dx == 'nv':
                color = [random.randint(20, 50), random.randint(40, 70), random.randint(70, 100)] # Brown
            elif dx == 'mel':
                color = [random.randint(0, 30), random.randint(0, 30), random.randint(30, 60)] # Dark brown/black
            elif dx == 'vasc':
                color = [random.randint(0, 20), random.randint(0, 20), random.randint(150, 200)] # Red/Vascular
            else:
                color = [random.randint(30, 80), random.randint(50, 100), random.randint(80, 130)]
            
            # Draw spot
            cv2.circle(img, (center_x, center_y), radius, color, -1)
            
            # Add some noise/blur to simulate realistic lesions
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Save half to folder 1, half to folder 2
            target_folder = part1_dir if i < num_records // 2 else part2_dir
            image_path = target_folder / f"{image_id}.jpg"
            
            cv2.imwrite(str(image_path), img)
            
    # Also add 2 corrupted/empty files to test loader validation
    corrupted_id1 = "ISIC_0099999"
    corrupted_id2 = "ISIC_0088888"
    
    with open(part1_dir / f"{corrupted_id1}.jpg", "w") as f:
        f.write("corrupted file content")
        
    # Introduce one missing file (present in metadata but not in directory)
    # We add it to metadata in a append operation
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['HAM_0009999', corrupted_id2, 'mel', 'histo', 50.0, 'male', 'back'])
        
    print(f"Generated {num_records} valid synthetic image files and metadata records.")
    print(f"Added 1 corrupted file and 1 missing file for validation testing.")

if __name__ == "__main__":
    setup_demo_data(Path("."))
