import os
import sys
import unittest
import json
from pathlib import Path
import pandas as pd

# Configure path variables to import backend and src modules
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from app import create_app

class TestFlaskAPI(unittest.TestCase):
    """
    Automated API unit test suite.
    Tests health endpoint, valid predictions, missing files, and corrupted files.
    """
    @classmethod
    def setUpClass(cls):
        # Initialize app with standard model and label encoder
        cls.app = create_app(
            model_path="models/efficientnet_b0_best.keras",
            encoder_path="processed/label_encoder.json"
        )
        cls.client = cls.app.test_client()
        
        # Setup directories
        cls.processed_dir = root_dir / "processed"
        
        # 1. Dynamically find a valid test image on disk
        cls.valid_image_path = None
        test_csv_path = cls.processed_dir / "test.csv"
        if test_csv_path.exists():
            df = pd.read_csv(test_csv_path)
            for path in df['image_path'].dropna():
                if os.path.exists(path):
                    cls.valid_image_path = Path(path)
                    break
                    
        # 2. Create unsupported format test file
        cls.unsupported_file_path = root_dir / "processed" / "test_unsupported.txt"
        with open(cls.unsupported_file_path, "w") as f:
            f.write("this is a text document, not a skin lesion image.")
            
        # 3. Create corrupted image test file
        cls.corrupted_file_path = root_dir / "processed" / "test_corrupted.jpg"
        with open(cls.corrupted_file_path, "w") as f:
            f.write("corrupted image bytes that fail opencv read check.")
            
    @classmethod
    def tearDownClass(cls):
        # Clean up temporary test files
        if cls.unsupported_file_path.exists():
            os.remove(cls.unsupported_file_path)
        if cls.corrupted_file_path.exists():
            os.remove(cls.corrupted_file_path)
            
    def test_health_endpoint(self):
        """Tests GET /api/health endpoint structure and status code."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("model_loaded"))
        
    def test_predict_endpoint_valid(self):
        """Tests POST /api/predict with a valid lesion image."""
        if not self.valid_image_path:
            self.skipTest("No valid test images found on disk to run prediction test.")
            
        with open(self.valid_image_path, 'rb') as img:
            data = {
                'file': (img, self.valid_image_path.name)
            }
            response = self.client.post(
                '/api/predict',
                data=data,
                content_type='multipart/form-data'
            )
            
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data.decode('utf-8'))
        self.assertIn("predicted_class", res_data)
        self.assertIn("confidence", res_data)
        self.assertIn("top_predictions", res_data)
        self.assertEqual(len(res_data["top_predictions"]), 3)
        
    def test_predict_endpoint_missing_file(self):
        """Tests POST /api/predict with an empty form payload (missing file)."""
        response = self.client.post(
            '/api/predict',
            data={},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 400)
        res_data = json.loads(response.data.decode('utf-8'))
        self.assertIn("error", res_data)
        self.assertEqual(res_data["error"], "No file part in the request")
        
    def test_predict_endpoint_unsupported_format(self):
        """Tests POST /api/predict with an unsupported file extension."""
        with open(self.unsupported_file_path, 'rb') as f:
            data = {
                'file': (f, self.unsupported_file_path.name)
            }
            response = self.client.post(
                '/api/predict',
                data=data,
                content_type='multipart/form-data'
            )
            
        self.assertEqual(response.status_code, 400)
        res_data = json.loads(response.data.decode('utf-8'))
        self.assertIn("error", res_data)
        self.assertTrue("Unsupported file format" in res_data["error"])
        
    def test_predict_endpoint_corrupted_image(self):
        """Tests POST /api/predict with a corrupted binary payload."""
        with open(self.corrupted_file_path, 'rb') as f:
            data = {
                'file': (f, self.corrupted_file_path.name)
            }
            response = self.client.post(
                '/api/predict',
                data=data,
                content_type='multipart/form-data'
            )
            
        self.assertEqual(response.status_code, 400)
        res_data = json.loads(response.data.decode('utf-8'))
        self.assertIn("error", res_data)
        self.assertTrue("Corrupted image file" in res_data["error"])

if __name__ == "__main__":
    unittest.main()
