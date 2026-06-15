import os
import sys
import time
import urllib.request
import urllib.error
import json
from pathlib import Path

URL_PREDICT = "http://127.0.0.1:5000/api/predict"

def print_banner(text):
    print("\n" + "=" * 70)
    print(f" {text.upper()}")
    print("=" * 70)

def build_multipart_body(filename, file_bytes, content_type="image/jpeg"):
    boundary = "OODVerificationBoundary98765"
    body_parts = []
    body_parts.append(f"--{boundary}")
    body_parts.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"')
    body_parts.append(f"Content-Type: {content_type}")
    body_parts.append("")
    body_parts.append(file_bytes)
    body_parts.append(f"--{boundary}--")
    body_parts.append("")
    
    binary_body = b""
    for part in body_parts:
        if isinstance(part, str):
            binary_body += part.encode('utf-8') + b"\r\n"
        else:
            binary_body += part + b"\r\n"
            
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(binary_body))
    }
    return binary_body, headers

def send_upload(image_path_or_bytes, filename=None):
    if isinstance(image_path_or_bytes, Path) or isinstance(image_path_or_bytes, str):
        path = Path(image_path_or_bytes)
        if not path.exists():
            return None, 404, {"error": "File not found locally"}
        file_bytes = path.read_bytes()
        fname = filename or path.name
    else:
        file_bytes = image_path_or_bytes
        fname = filename or "corrupt.jpg"

    data, headers = build_multipart_body(fname, file_bytes)
    req = urllib.request.Request(URL_PREDICT, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode('utf-8'))
            return resp.status, body
    except urllib.error.HTTPError as err:
        try:
            body = json.loads(err.read().decode('utf-8'))
        except Exception:
            body = {"error": err.reason}
        return err.code, body
    except Exception as e:
        return 500, {"error": str(e)}

def main():
    print_banner("Sprint 11A - Out-of-Distribution & Input Quality Verification")
    
    test_dir = Path("test_validation")
    if not test_dir.exists():
        print("Error: test_validation/ directory does not exist. Run generation script first.")
        sys.exit(1)

    # Test cases mapping: file_path/bytes -> (expected_success, expected_status, test_name, expected_message_part)
    test_cases = [
        # 1. Valid lesion image (accepted or marked uncertain)
        (test_dir / "lesion_ISIC_0024306.jpg", True, 200, "Valid Skin Lesion Image", None),
        (test_dir / "lesion_low_conf.jpg", True, 200, "Low-Confidence Lesion Image", None),
        
        # 2. OOD images (rejected)
        (test_dir / "cat_1.jpg", False, 400, "Cat Image (OOD)", "Please upload a dermoscopic skin lesion image"),
        (test_dir / "car_1.jpg", False, 400, "Car Image (OOD)", "Please upload a dermoscopic skin lesion image"),
        (test_dir / "building_1.jpg", False, 400, "Building Image (OOD)", "Please upload a dermoscopic skin lesion image"),
        (test_dir / "face_1.jpg", False, 400, "Human Face Image (OOD)", "Please upload a dermoscopic skin lesion image"),
        (test_dir / "food_1.jpg", False, 400, "Food Image (OOD)", "Please upload a dermoscopic skin lesion image"),
        
        # 3. Quality defects (rejected)
        (test_dir / "empty_flat.jpg", False, 400, "Empty Flat-Color Image", "empty or contains solid flat color blocks"),
        (test_dir / "blurry_lesion.jpg", False, 400, "Blurry Lesion Image", "too blurry"),
        (test_dir / "low_res_lesion.jpg", False, 400, "Low-Resolution Lesion Image", "too low"),
        (test_dir / "dark_lesion.jpg", False, 400, "Extremely Dark Lesion Image", "extremely dark"),
        (test_dir / "bright_lesion.jpg", False, 400, "Extremely Bright Lesion Image", "extremely bright"),
        
        # 4. Format & corruption checks (rejected)
        (b"corrupted raw data payload bytes", False, 400, "Corrupted Bytes Stream", "Failed to decode pixel arrays")
    ]

    all_passed = True
    results = []

    print(f"{'Test Case':<32} | {'Expected':<12} | {'Got':<12} | {'Result':<10}")
    print("-" * 75)

    for path_or_bytes, is_valid, exp_code, name, msg_part in test_cases:
        code, body = send_upload(path_or_bytes)
        
        # Determine if test case matches assertions
        passed = False
        if is_valid:
            # For valid lesions, it should return 200 and not have a status of "rejected"
            passed = (code == 200 and body.get("status") != "rejected")
        else:
            # For invalid cases, we expect a 400 code and "rejected" status or specific format error
            if code == 400:
                if msg_part:
                    passed = (msg_part.lower() in body.get("message", "").lower() or msg_part.lower() in body.get("error", "").lower())
                else:
                    passed = True

        status_text = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        if not passed:
            all_passed = False

        expected_desc = "Accept" if is_valid else "Reject (400)"
        got_desc = f"{code} OK" if (code == 200) else f"{code} Fail"
        
        print(f"{name:<32} | {expected_desc:<12} | {got_desc:<12} | {status_text:<10}")
        
        results.append({
            "name": name,
            "expected_valid": is_valid,
            "actual_code": code,
            "passed": passed,
            "response": body
        })

    # Test confidence threshold trigger by making a request where predicted confidence < 0.70
    print_banner("Confidence Threshold Check")
    # Locate the low confidence lesion image results
    low_conf_res = None
    for res in results:
        if "Low-Confidence Lesion" in res["name"]:
            low_conf_res = res
            break
            
    if low_conf_res:
        body = low_conf_res["response"]
        print(f"Low-confidence lesion response: {body}")
        is_uncertain = (body.get("status") == "uncertain" and "confidence is too low" in body.get("message", "").lower())
        if is_uncertain:
            print("\033[92m[PASS]\033[0m Successfully triggered classification threshold (status is 'uncertain')")
        else:
            print("\033[91m[FAIL]\033[0m Failed to trigger classification threshold on low_conf request.")
            all_passed = False
    else:
        print("\033[91m[FAIL]\033[0m Low-confidence lesion test results not found.")
        all_passed = False

    # Save summary log to reports
    log_payload = {
        "timestamp": time.time(),
        "all_passed": all_passed,
        "results": results
    }
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "ood_raw_results.json", "w") as f:
        json.dump(log_payload, f, indent=4)

    print_banner("Summary Verification Verdict")
    if all_passed:
        print("\033[92mOOD VERIFICATION COMPLETED: ALL TESTS PASSED SUCCESSFULLY!\033[0m")
        sys.exit(0)
    else:
        print("\033[91mOOD VERIFICATION FAILED: NOT ALL ASSERTIONS PASSED.\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()
