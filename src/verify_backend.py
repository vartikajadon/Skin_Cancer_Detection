import os
import sys
import unittest
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_verification():
    logger.info("Initializing Sprint 8 Backend Verification Suite...")
    
    base_dir = Path(".")
    app_py = base_dir / "backend" / "app.py"
    routes_py = base_dir / "backend" / "routes.py"
    pred_svc = base_dir / "backend" / "services" / "prediction_service.py"
    img_val = base_dir / "backend" / "utils" / "image_validator.py"
    
    test_api = base_dir / "tests" / "test_api.py"
    
    api_doc = base_dir / "reports" / "api_documentation.md"
    readiness_report = base_dir / "reports" / "backend_readiness_report.md"
    
    # 1. Verify existence of backend files
    assert app_py.exists(), "app.py missing"
    assert routes_py.exists(), "routes.py missing"
    assert pred_svc.exists(), "prediction_service.py missing"
    assert img_val.exists(), "image_validator.py missing"
    logger.info("✅ Verification Pass: All Flask backend modules exist.")
    
    # 2. Verify existence of test suite
    assert test_api.exists(), "test_api.py missing"
    logger.info("✅ Verification Pass: API unit test suite exists.")
    
    # 3. Verify existence of reports
    assert api_doc.exists(), "api_documentation.md missing"
    assert readiness_report.exists(), "backend_readiness_report.md missing"
    logger.info("✅ Verification Pass: API documentation and readiness reports generated.")
    
    # 4. Programmatically run unittest suite
    logger.info("Running API unit test suite programmatically...")
    sys.path.insert(0, str(base_dir / "tests"))
    from test_api import TestFlaskAPI
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFlaskAPI)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    assert result.wasSuccessful(), "Verification failed: API unit tests failed."
    logger.info("✅ Verification Pass: All 5 API unit tests passed successfully.")
    
    logger.info("🎉 All Sprint 8 Flask Backend API Verification checks passed successfully!")

if __name__ == "__main__":
    run_verification()
