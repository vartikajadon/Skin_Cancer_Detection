import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

# Setup path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from predict import LesionPredictor
from gradcam import GradCAMGenerator

def create_collage():
    print("Initializing predictor modules...")
    # 1. Initialize predictor
    predictor = LesionPredictor(
        Path("models/efficientnet_b0_best.keras"),
        Path("processed/label_encoder.json")
    )
    generator = GradCAMGenerator(predictor.model)
    
    # 2. Find 3 unique class images in test dataset
    test_csv = Path("processed/test.csv")
    if not test_csv.exists():
        print(f"Error: Missing test dataset metadata at {test_csv}")
        return
        
    df = pd.read_csv(test_csv)
    image_paths = []
    
    # Group by label to pull unique class instances
    grouped = df.groupby('label')
    for label, group in grouped:
        for p in group['image_path'].dropna():
            if os.path.exists(p):
                image_paths.append(Path(p))
                break
        if len(image_paths) >= 3:
            break
            
    # Fallback to general listing if unique count fails
    if len(image_paths) < 3:
        image_paths = []
        for p in df['image_path'].dropna():
            if os.path.exists(p):
                image_paths.append(Path(p))
                if len(image_paths) >= 3:
                    break
                    
    if len(image_paths) < 3:
        print("Error: Could not locate at least 3 test images on disk.")
        return
        
    print(f"Loaded integration test files: {[p.name for p in image_paths]}")
    
    # Grid sizes
    img_size = 224
    padding = 10
    header_height = 40
    title_height = 50
    
    width = 3 * img_size + 4 * padding
    height = title_height + 3 * (img_size + header_height) + padding
    
    # Create blank canvas
    collage = Image.new("RGB", (width, height), color=(244, 247, 251))
    draw = ImageDraw.Draw(collage)
    
    # Draw Title text (Draw line-based indicator fallback since fonts are environment specific)
    draw.text((padding * 2, 15), "Grad-CAM Neural Attention Maps - EfficientNetB0", fill=(15, 98, 254))
    draw.line([(padding * 2, 35), (width - padding * 2, 35)], fill=(15, 98, 254), width=1)
    
    class_labels = {
        'nv': 'Nevus (Benign Mole)',
        'mel': 'Melanoma (Malignant Lesion)',
        'bkl': 'Benign Keratosis-like Lesion',
        'bcc': 'Basal Cell Carcinoma (Suspected)',
        'akiec': 'Actinic Keratosis / Bowen\'s Disease',
        'vasc': 'Vascular Lesion',
        'df': 'Dermatofibroma'
    }

    # Generate grid contents
    for row_idx, img_path in enumerate(image_paths):
        # 1. Run predictions
        res = predictor.predict_image(img_path)
        pred_class = res["predicted_class"]
        confidence = res["confidence"]
        label_str = class_labels.get(pred_class, pred_class)
        
        # 2. Run Grad-CAM
        class_idx = predictor.encoder_map.get(pred_class)
        explain_res = generator.generate_heatmap_and_overlay(img_path, class_idx)
        
        # 3. Read generated outputs
        orig_img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
        heat_img = Image.open(ROOT_DIR / explain_res["heatmap_path"]).convert("RGB").resize((img_size, img_size))
        over_img = Image.open(ROOT_DIR / explain_res["overlay_path"]).convert("RGB").resize((img_size, img_size))
        
        # Draw row descriptors
        row_y_start = title_height + row_idx * (img_size + header_height)
        row_text = f"Case {row_idx + 1}: {label_str} (Confidence Score: {confidence * 100:.1f}%)"
        draw.text((padding * 2, row_y_start + 10), row_text, fill=(22, 22, 22))
        
        img_y = row_y_start + header_height
        
        # Paste row visuals
        collage.paste(orig_img, (padding, img_y))
        collage.paste(heat_img, (2 * padding + img_size, img_y))
        collage.paste(over_img, (3 * padding + 2 * img_size, img_y))
        
        # Draw column labels on first row
        if row_idx == 0:
            draw.text((padding, img_y - 15), "Original Lesion", fill=(111, 111, 111))
            draw.text((2 * padding + img_size, img_y - 15), "Attention Heatmap", fill=(111, 111, 111))
            draw.text((3 * padding + 2 * img_size, img_y - 15), "Superimposed Overlay", fill=(111, 111, 111))
            
    # Save collage
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    collage.save(reports_dir / "gradcam_examples.png")
    print(f"Grad-CAM examples collage saved successfully to {reports_dir / 'gradcam_examples.png'}")

if __name__ == "__main__":
    create_collage()
