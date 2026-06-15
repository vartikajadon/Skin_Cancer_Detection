import os
import argparse
import logging
from pathlib import Path
import pandas as pd
from predict import LesionPredictor, InferenceError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchPredictor:
    """
    Coordinates batch-level inference across image lists or directories.
    Handles individual failures gracefully to prevent batch pipeline interruptions.
    """
    def __init__(self, predictor: LesionPredictor):
        self.predictor = predictor
        
    def predict_batch_from_paths(self, image_paths: list) -> Tuple[pd.DataFrame, dict]:
        """
        Runs inference on a list of image paths.
        Returns:
            DataFrame containing predictions.
            Dict containing batch statistics.
        """
        logger.info(f"Starting batch prediction on {len(image_paths)} image files...")
        
        results = []
        success_count = 0
        failure_count = 0
        confidences = []
        class_counts = {}
        
        for idx, path in enumerate(image_paths):
            path = Path(path)
            record = {
                "image_name": path.name,
                "image_id": path.stem,
                "predicted_class": "",
                "confidence": 0.0,
                "error": ""
            }
            
            try:
                pred = self.predictor.predict_image(path)
                
                record["predicted_class"] = pred["predicted_class"]
                record["confidence"] = pred["confidence"]
                success_count += 1
                confidences.append(pred["confidence"])
                
                class_counts[pred["predicted_class"]] = class_counts.get(pred["predicted_class"], 0) + 1
                
            except Exception as e:
                # Capture individual file exception details
                record["error"] = str(e)
                failure_count += 1
                logger.warning(f"Skipping file {path.name}: {str(e)}")
                
            results.append(record)
            
        # Compile results DataFrame
        df_results = pd.DataFrame(results)
        
        # Calculate stats
        stats = {
            "total_processed": len(image_paths),
            "successful_predictions": success_count,
            "failed_predictions": failure_count,
            "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "class_distribution": class_counts
        }
        
        logger.info(f"Batch prediction complete. Success: {success_count}, Failures: {failure_count}")
        return df_results, stats

def main():
    base_dir = Path(".")
    model_path = base_dir / "models" / "efficientnet_b0_best.keras"
    encoder_path = base_dir / "processed" / "label_encoder.json"
    processed_dir = base_dir / "processed"
    
    parser = argparse.ArgumentParser(description="Skin Cancer Batch Prediction CLI")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--output", type=str, default="processed/predictions.csv", help="Path to save predictions CSV")
    args = parser.parse_args()
    
    try:
        # 1. Initialize Predictor
        predictor = LesionPredictor(model_path, encoder_path)
        batch_predictor = BatchPredictor(predictor)
        
        # 2. Collect image paths
        image_paths = []
        
        if args.image_dir:
            image_dir = Path(args.image_dir)
            if image_dir.exists() and image_dir.is_dir():
                valid_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
                image_paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix in valid_exts]
                logger.info(f"Scanned {len(image_paths)} images in input directory: {image_dir}")
        else:
            # Default fallback: Load test split from CSV and use paths present on disk
            test_csv_path = processed_dir / "test.csv"
            if test_csv_path.exists():
                logger.info(f"No image directory specified. Loading disk-verified test split from {test_csv_path}...")
                test_df = pd.read_csv(test_csv_path)
                test_df = test_df[test_df['image_path'].notna()]
                image_paths = [Path(p) for p in test_df['image_path'] if os.path.exists(p)]
                logger.info(f"Collected {len(image_paths)} disk-verified test images.")
                
        if not image_paths:
            logger.error("No valid image files found to process.")
            return
            
        # 3. Add intentionally corrupted/unsupported files to batch to verify robust error handling
        # This acts as a pipeline sanity-check stochastically
        logger.info("Injecting test cases to verify pipeline error handling...")
        # Create a non-existent path
        image_paths.append(base_dir / "data" / "non_existent_lesion.jpg")
        # Create an unsupported format path
        image_paths.append(base_dir / "data" / "unsupported_doc.pdf")
        # Create a corrupted image file path (we'll use the corrupted file generated in demo_setup if present, or create a mock file)
        corrupted_path = base_dir / "data" / "HAM10000_images_part_1" / "ISIC_0099999.jpg"
        if corrupted_path.exists():
            image_paths.append(corrupted_path)
        else:
            # Create a mock corrupted text file masquerading as jpg
            mock_corrupt = base_dir / "processed" / "temp_corrupt_test.jpg"
            with open(mock_corrupt, "w") as f:
                f.write("this is corrupt image pixel array")
            image_paths.append(mock_corrupt)
            
        # 4. Run Batch Inference
        df_results, stats = batch_predictor.predict_batch_from_paths(image_paths)
        
        # Clean up mock corrupt file if created
        mock_corrupt_temp = base_dir / "processed" / "temp_corrupt_test.jpg"
        if mock_corrupt_temp.exists():
            os.remove(mock_corrupt_temp)
            
        # 5. Export results
        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_csv, index=False)
        logger.info(f"Saved batch prediction results to {output_csv.resolve()}")
        
        # Print stats
        print("\n=== BATCH RUN STATISTICS ===")
        print(f"Total Processed: {stats['total_processed']}")
        print(f"Success Count:   {stats['successful_predictions']}")
        print(f"Failure Count:   {stats['failed_predictions']}")
        print(f"Mean Confidence: {stats['average_confidence']:.4f}")
        print("Class Predictions Distribution:")
        for k, v in stats['class_distribution'].items():
            print(f"  - {k}: {v}")
            
    except Exception as e:
        logger.error(f"Batch inference failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Import Tuple locally to ensure standard script structure
    from typing import Tuple
    import numpy as np
    main()
