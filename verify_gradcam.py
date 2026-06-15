import os
import sys
import time
import subprocess
import urllib.request
import urllib.error
import json
import re
from pathlib import Path
import cv2

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_DIR = ROOT_DIR / "backend"
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

def print_result(step_name, success, message=""):
    status = "[SUCCESS]" if success else "[FAILED]"
    color_code = "\033[92m" if success else "\033[91m"
    reset_code = "\033[0m"
    print(f"{color_code}{status} {step_name}{reset_code} {message}")

def locate_test_image():
    test_csv = Path("processed/test.csv")
    if test_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(test_csv)
            for path in df['image_path'].dropna():
                if os.path.exists(path):
                    return Path(path)
        except Exception:
            pass
    for p in Path("data").rglob("*.jpg"):
        if p.exists():
            return p
    for p in Path("processed").rglob("*.jpg"):
        if p.exists():
            return p
    return None

def verify_gradcam_generator_unit():
    print("\n--- Running GradCAMGenerator Unit Tests ---")
    try:
        from gradcam import GradCAMGenerator
        from predict import LesionPredictor
        
        # 1. Initialize predictor
        predictor = LesionPredictor(
            Path("models/efficientnet_b0_best.keras"),
            Path("processed/label_encoder.json")
        )
        generator = GradCAMGenerator(predictor.model)
        print_result("Generator instantiation", True, "GradCAMGenerator loaded successfully")

        # 2. Locate test image
        valid_image = locate_test_image()
        if not valid_image:
            print_result("Locate test image", False, "Could not find a valid test image on disk")
            return False

        # 3. Generate heatmap & overlay
        results = generator.generate_heatmap_and_overlay(valid_image)
        
        # Verify file existence
        heatmap_path = ROOT_DIR / results["heatmap_path"]
        overlay_path = ROOT_DIR / results["overlay_path"]
        
        assert heatmap_path.exists(), "heatmap.png does not exist"
        assert overlay_path.exists(), "overlay.png does not exist"
        print_result("Generate explainability outputs", True, f"Saved heatmap and overlay to disk")

        # 4. Verify image dimensions
        h_img = cv2.imread(str(heatmap_path))
        o_img = cv2.imread(str(overlay_path))
        
        assert h_img.shape == (224, 224, 3), f"Expected heatmap shape (224, 224, 3), got {h_img.shape}"
        assert o_img.shape == (224, 224, 3), f"Expected overlay shape (224, 224, 3), got {o_img.shape}"
        print_result("Outputs shape validation", True, "Images resized correctly to 224x224x3")

        # 5. Verify base64 strings
        assert results["heatmap_base64"].startswith("data:image/png;base64,"), "Invalid heatmap base64 format"
        assert results["overlay_base64"].startswith("data:image/png;base64,"), "Invalid overlay base64 format"
        print_result("Outputs base64 validation", True, "Base64 data strings formatted correctly")
        
        return True
    except Exception as e:
        print_result("Generator unit test exception", False, str(e))
        return False

def verify_frontend_gradcam():
    print("\n--- Verifying Frontend Grad-CAM Integrations ---")
    
    # 1. Check index.html elements
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        print_result("index.html exists", False, "Missing frontend/index.html")
        return False
        
    html_content = html_path.read_text(encoding="utf-8")
    elements = ['id="gradcam-container"', 'id="gradcam-original"', 'id="gradcam-heatmap"', 'id="gradcam-overlay"']
    missing_els = [el for el in elements if el not in html_content]
    
    if missing_els:
        print_result("HTML elements check", False, f"Missing elements in index.html: {', '.join(missing_els)}")
        return False
    print_result("HTML elements verification", True, "All Grad-CAM image display tags are present")

    # 2. Check app.js logic mapping
    app_js_path = FRONTEND_DIR / "js" / "app.js"
    if not app_js_path.exists():
        print_result("app.js exists", False, "Missing frontend/js/app.js")
        return False
        
    app_content = app_js_path.read_text(encoding="utf-8")
    js_refs = ["gradcamContainer", "gradcamOriginal", "gradcamHeatmap", "gradcamOverlay", "result.gradcam_image_base64"]
    missing_refs = [ref for ref in js_refs if ref not in app_content]
    
    if missing_refs:
        print_result("JS event mapping check", False, f"Missing bindings in app.js: {', '.join(missing_refs)}")
        return False
    print_result("JS event mapping verification", True, "app.js successfully maps and binds explainability payloads")

    # 3. Check style.css rules
    css_path = FRONTEND_DIR / "css" / "style.css"
    if not css_path.exists():
        print_result("style.css exists", False, "Missing style.css")
        return False
        
    css_content = css_path.read_text(encoding="utf-8")
    css_classes = [".gradcam-container", ".gradcam-grid", ".gradcam-card", ".gradcam-img", ".gradcam-explanation-text"]
    missing_cls = [c for c in css_classes if c not in css_content]
    
    if missing_cls:
        print_result("CSS class styles check", False, f"Missing classes in style.css: {', '.join(missing_cls)}")
        return False
    print_result("CSS class styles verification", True, "All explainability layout rules are present")

    return True

def verify_end_to_end_api():
    print("\n--- Running End-to-End API Grad-CAM Communication Tests ---")
    
    # Locate test image
    valid_image = locate_test_image()
    if not valid_image:
        print_result("Locate test image", False, "No test image to execute API checks")
        return False

    # Launch local Flask server in background
    print("Launching Flask application server in background...")
    server_process = subprocess.Popen(
        [sys.executable, str(BACKEND_DIR / "app.py")],
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    url_health = "http://127.0.0.1:5000/api/health"
    url_predict = "http://127.0.0.1:5000/api/predict"
    server_up = False
    
    # Wait for server
    for i in range(25):
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
        return False
        
    print_result("server launch", True, "Local Flask server is online")

    test_success = True
    try:
        # Build multipart post
        boundary = "GradCAMBoundary123"
        body_parts = []
        body_parts.append(f"--{boundary}")
        body_parts.append(f'Content-Disposition: form-data; name="file"; filename="{valid_image.name}"')
        body_parts.append("Content-Type: image/jpeg")
        body_parts.append("")
        body_parts.append(valid_image.read_bytes())
        body_parts.append(f"--{boundary}--")
        body_parts.append("")
        
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
        
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            
            # Assert explainability outputs are present in response JSON
            assert "predicted_class" in data
            assert "confidence" in data
            assert data.get("gradcam_image") == "overlay.png", f"Expected 'overlay.png', got {data.get('gradcam_image')}"
            assert data.get("heatmap_image") == "heatmap.png", f"Expected 'heatmap.png', got {data.get('heatmap_image')}"
            assert data.get("gradcam_image_base64").startswith("data:image/png;base64,"), "Missing overlay base64"
            assert data.get("heatmap_image_base64").startswith("data:image/png;base64,"), "Missing heatmap base64"
            
            print_result("API Predict extended response structure", True, "Correctly exposes explainability image paths and base64 payloads")
            
        # Assert files are available statically from server
        url_static_overlay = "http://127.0.0.1:5000/gradcam_outputs/overlay.png"
        with urllib.request.urlopen(url_static_overlay) as static_resp:
            assert static_resp.status == 200, "Static overlay.png serve check failed"
            assert len(static_resp.read()) > 1000, "Static served image is empty"
            print_result("API serve static assets", True, "Serving generated files statically from /gradcam_outputs/ successfully")

    except Exception as e:
        print_result("API explainability validation exception", False, str(e))
        test_success = False
    finally:
        print("Stopping background Flask server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            server_process.kill()
            
    return test_success

def main():
    print("======================================================================")
    print("           SKIN CANCER DETECTION - GRAD-CAM VERIFICATION SUITE        ")
    print("======================================================================")
    
    unit_ok = verify_gradcam_generator_unit()
    if not unit_ok:
        print("\033[91mCRITICAL: GradCAMGenerator unit tests failed.\033[0m")
        sys.exit(1)
        
    frontend_ok = verify_frontend_gradcam()
    if not frontend_ok:
        print("\033[91mCRITICAL: Frontend Grad-CAM files alignment checks failed.\033[0m")
        sys.exit(1)
        
    api_ok = verify_end_to_end_api()
    if not api_ok:
        print("\033[91mCRITICAL: API explainability integrations failed.\033[0m")
        sys.exit(1)
        
    print("\n\033[92mGRAD-CAM VERIFICATION COMPLETE: ALL CHECKS PASSED SUCCESSFULLY!\033[0m")
    sys.exit(0)

if __name__ == "__main__":
    main()
