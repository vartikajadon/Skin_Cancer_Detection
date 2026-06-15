import os
import sys
import time
import subprocess
import urllib.request
import urllib.error
import json
import re
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_DIR = ROOT_DIR / "backend"
TESTS_DIR = ROOT_DIR / "tests"

def print_result(step_name, success, message=""):
    status = "[SUCCESS]" if success else "[FAILED]"
    color_code = "\033[92m" if success else "\033[91m"
    reset_code = "\033[0m"
    print(f"{color_code}{status} {step_name}{reset_code} {message}")

def verify_frontend_files():
    print("\n--- Verifying Frontend File Structure & Inclusions ---")
    
    # 1. Check config.js
    config_path = FRONTEND_DIR / "config.js"
    if not config_path.exists():
        print_result("config.js exists", False, "Missing frontend/config.js")
        return False
    
    config_content = config_path.read_text(encoding="utf-8")
    if "API_BASE_URL" not in config_content:
        print_result("config.js configuration", False, "API_BASE_URL not defined in config.js")
        return False
    print_result("config.js setup", True, "frontend/config.js is valid and defines API_BASE_URL")
    
    # 2. Check api.js
    api_js_path = FRONTEND_DIR / "js" / "api.js"
    if not api_js_path.exists():
        print_result("api.js exists", False, "Missing frontend/js/api.js")
        return False
    
    api_content = api_js_path.read_text(encoding="utf-8")
    if "predictLesion" not in api_content or "healthCheck" not in api_content:
        print_result("api.js structure", False, "predictLesion or healthCheck functions missing in api.js")
        return False
    print_result("api.js setup", True, "frontend/js/api.js is valid and exposes API service wrappers")

    # 3. Check HTML Script Tag references
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        print_result("index.html exists", False, "Missing frontend/index.html")
        return False
        
    html_content = html_path.read_text(encoding="utf-8")
    
    # Check for correct script order: config.js -> api.js -> app.js
    scripts_pattern = r'<script src="config\.js"></script>.*?<script src="js/api\.js"></script>.*?<script src="js/app\.js"></script>'
    has_correct_scripts = re.search(scripts_pattern, html_content, re.DOTALL) is not None
    
    if not has_correct_scripts:
        print_result("index.html script order", False, 
                     "Script elements are missing or loaded in the incorrect order. Expected: config.js -> js/api.js -> js/app.js")
        return False
        
    # Check for top-predictions-list container
    if 'id="top-predictions-list"' not in html_content:
        print_result("index.html elements", False, "Missing #top-predictions-list container inside index.html")
        return False
        
    # Check for success notification container
    if 'id="success-notification"' not in html_content:
        print_result("index.html elements", False, "Missing #success-notification element inside index.html")
        return False

    print_result("index.html structure", True, "Inclusions and DOM container configurations are valid")
    
    # 4. Check CSS style classes
    css_path = FRONTEND_DIR / "css" / "style.css"
    if not css_path.exists():
        print_result("style.css exists", False, "Missing frontend/css/style.css")
        return False
        
    css_content = css_path.read_text(encoding="utf-8")
    css_classes = [".risk-badge", ".spinner-loader", ".error-alert-card", ".top-pred-fill", ".success-notification"]
    missing_classes = [cls for cls in css_classes if cls not in css_content]
    
    if missing_classes:
        print_result("style.css classes check", False, f"Missing classes in style.css: {', '.join(missing_classes)}")
        return False
        
    print_result("style.css updates", True, "All integration style classes are present")
    return True

def run_integration_tests():
    print("\n--- Running End-to-End API Integration Communication Tests ---")
    
    # 1. Setup sample test files
    unsupported_txt = ROOT_DIR / "processed" / "temp_integration_unsupported.txt"
    unsupported_txt.write_text("not an image file content", encoding="utf-8")
    
    # Find a valid test image using processed test metadata
    valid_image_path = None
    test_csv = ROOT_DIR / "processed" / "test.csv"
    if test_csv.exists():
        try:
            with open(test_csv, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Parse image_path column (index depends on layout, we look for image paths)
                    cols = lines[0].strip().split(',')
                    path_idx = -1
                    for i, col in enumerate(cols):
                        if 'path' in col.lower():
                            path_idx = i
                            break
                    if path_idx != -1:
                        for line in lines[1:]:
                            val = line.strip().split(',')[path_idx]
                            val = val.replace('"', '').replace("'", "")
                            if os.path.exists(val):
                                valid_image_path = Path(val)
                                break
        except Exception as e:
            print(f"Warning parsing test.csv: {e}")
            
    # Fallback search in processed directory
    if not valid_image_path:
        for p in (ROOT_DIR / "processed").rglob("*.jpg"):
            valid_image_path = p
            break
            
    if not valid_image_path:
        print_result("locate test image", False, "Could not find a valid test image to execute integration tests.")
        if unsupported_txt.exists():
            os.remove(unsupported_txt)
        return False
        
    print(f"Selected valid integration test image: {valid_image_path.name}")
    
    # 2. Launch local Flask server in background process
    print("Launching Flask application server in background...")
    server_process = subprocess.Popen(
        [sys.executable, str(BACKEND_DIR / "app.py")],
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to boot up
    url_health = "http://127.0.0.1:5000/api/health"
    url_predict = "http://127.0.0.1:5000/api/predict"
    server_up = False
    
    for i in range(25): # Wait up to 5 seconds
        time.sleep(0.2)
        try:
            with urllib.request.urlopen(url_health, timeout=1.0) as resp:
                if resp.status == 200:
                    server_up = True
                    break
        except Exception:
            pass
            
    if not server_up:
        print_result("server launch", False, "Flask server failed to boot up within 5 seconds.")
        server_process.terminate()
        if unsupported_txt.exists():
            os.remove(unsupported_txt)
        return False
        
    print_result("server launch", True, "Local Flask server is online and responding to health checks")
    
    integration_success = True
    
    try:
        # Test 1: GET /api/health
        with urllib.request.urlopen(url_health) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            assert data.get("status") == "healthy", "Expected 'healthy' status"
            assert data.get("model_loaded") is True, "Expected model_loaded to be True"
            print_result("GET /api/health response validation", True, f"Response: {data}")
            
        # Test 2: POST /api/predict (Valid Lesion Image)
        # Construct Multipart Form Post manually without dependencies to ensure standalone run
        boundary = "IntegrationTestBoundary12345"
        body_parts = []
        
        # Add file
        body_parts.append(f"--{boundary}")
        body_parts.append(f'Content-Disposition: form-data; name="file"; filename="{valid_image_path.name}"')
        body_parts.append("Content-Type: image/jpeg")
        body_parts.append("")
        with open(valid_image_path, "rb") as f:
            body_parts.append(f.read())
            
        body_parts.append(f"--{boundary}--")
        body_parts.append("")
        
        # Build binary body payload
        binary_body = b""
        for part in body_parts:
            if isinstance(part, str):
                binary_body += part.encode('utf-8') + b"\r\n"
            else:
                binary_body += part + b"\r\n"
                
        req = urllib.request.Request(
            url_predict,
            data=binary_body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(binary_body))
            },
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                assert "predicted_class" in data, "Missing 'predicted_class' in payload"
                assert "confidence" in data, "Missing 'confidence' in payload"
                assert "top_predictions" in data, "Missing 'top_predictions' in payload"
                assert len(data["top_predictions"]) == 3, f"Expected top 3 predictions, got {len(data['top_predictions'])}"
                print_result("POST /api/predict success response", True, f"Response: {data}")
        except urllib.error.HTTPError as err:
            err_body = err.read().decode('utf-8')
            print_result("POST /api/predict success response", False, f"HTTP Error {err.code}: {err_body}")
            raise err
            
        # Test 3: POST /api/predict (Unsupported Format)
        body_parts_txt = []
        body_parts_txt.append(f"--{boundary}")
        body_parts_txt.append(f'Content-Disposition: form-data; name="file"; filename="{unsupported_txt.name}"')
        body_parts_txt.append("Content-Type: text/plain")
        body_parts_txt.append("")
        body_parts_txt.append(unsupported_txt.read_bytes())
        body_parts_txt.append(f"--{boundary}--")
        body_parts_txt.append("")
        
        binary_body_txt = b""
        for part in body_parts_txt:
            if isinstance(part, str):
                binary_body_txt += part.encode('utf-8') + b"\r\n"
            else:
                binary_body_txt += part + b"\r\n"
                
        req_txt = urllib.request.Request(
            url_predict,
            data=binary_body_txt,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(binary_body_txt))
            },
            method="POST"
        )
        
        try:
            urllib.request.urlopen(req_txt)
            print_result("POST /api/predict error validation", False, "Expected 400 error for unsupported file, but request succeeded.")
            integration_success = False
        except urllib.error.HTTPError as err:
            if err.code == 400:
                err_data = json.loads(err.read().decode('utf-8'))
                assert "error" in err_data, "Expected error explanation message"
                print_result("POST /api/predict error validation", True, f"Correctly blocked with status 400. Message: {err_data}")
            else:
                print_result("POST /api/predict error validation", False, f"Expected 400 but got: {err.code}")
                integration_success = False
                
    except Exception as e:
        print_result("communication tests exception", False, f"Integration communication test failed: {e}")
        integration_success = False
        
    finally:
        # Shutdown server
        print("Stopping background Flask server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            server_process.kill()
            
        # Clean up files
        if unsupported_txt.exists():
            os.remove(unsupported_txt)
            
    return integration_success

def main():
    print("======================================================================")
    print("         SKIN CANCER DETECTION - INTEGRATION VERIFICATION SUITE       ")
    print("======================================================================")
    
    frontend_ok = verify_frontend_files()
    if not frontend_ok:
        print("\n\033[91mCRITICAL: Frontend file structures are invalid.\033[0m")
        sys.exit(1)
        
    integration_ok = run_integration_tests()
    if not integration_ok:
        print("\n\033[91mCRITICAL: End-to-end communication tests failed.\033[0m")
        sys.exit(1)
        
    print("\n\033[92mINTEGRATION VERIFICATION COMPLETE: ALL ASSERTS PASSED SUCCESSFULLY!\033[0m")
    sys.exit(0)

if __name__ == "__main__":
    main()
