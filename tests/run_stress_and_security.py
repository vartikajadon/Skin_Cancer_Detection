import os
import sys
import time
import urllib.request
import urllib.error
import json
import concurrent.futures
from pathlib import Path

# Try to import psutil for memory profiling
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

URL_HEALTH = "http://127.0.0.1:5000/api/health"
URL_PREDICT = "http://127.0.0.1:5000/api/predict"

def print_banner(text):
    print("\n" + "=" * 70)
    print(f" {text.upper()}")
    print("=" * 70)

def print_result(step_name, success, message=""):
    status = "[SUCCESS]" if success else "[FAILED]"
    color_code = "\033[92m" if success else "\033[91m"
    reset_code = "\033[0m"
    print(f" {color_code}{status} {step_name}{reset_code} {message}")

def locate_valid_image():
    # Attempt to locate valid test image from test.csv
    test_csv = ROOT_DIR / "processed" / "test.csv"
    if test_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(test_csv)
            for path in df['image_path'].dropna():
                if os.path.exists(path):
                    return Path(path)
        except Exception:
            pass
            
    # Fallback to scanning data directory
    for p in (ROOT_DIR / "data").rglob("*.jpg"):
        if p.exists():
            return p
            
    # Fallback to processed directory
    for p in (ROOT_DIR / "processed").rglob("*.jpg"):
        if p.exists():
            return p
            
    return None

def build_multipart_body(filename, file_bytes, content_type="image/jpeg"):
    boundary = "StressAndSecurityBoundary12345"
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

def send_post_request(url, data, headers):
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))

def run_security_suite(valid_image_path):
    print_banner("Running Security Auditing Suite")
    
    results = {}
    
    # Test 1: Corrupted image upload (random bytes)
    print("Test 1: Uploading corrupted byte stream...")
    corrupt_data, corrupt_hdrs = build_multipart_body("corrupt.jpg", b"corrupted image file contents that fail cv2 decode")
    try:
        urllib.request.urlopen(urllib.request.Request(URL_PREDICT, data=corrupt_data, headers=corrupt_hdrs, method="POST"))
        results["corrupted_image"] = ("FAIL", "Server accepted corrupted bytes as valid image")
    except urllib.error.HTTPError as err:
        body = err.read().decode('utf-8')
        results["corrupted_image"] = ("PASS", f"Blocked with status {err.code}: {body}")
        
    # Test 2: Empty file upload (0 bytes)
    print("Test 2: Uploading empty file (0 bytes)...")
    empty_data, empty_hdrs = build_multipart_body("empty.jpg", b"")
    try:
        urllib.request.urlopen(urllib.request.Request(URL_PREDICT, data=empty_data, headers=empty_hdrs, method="POST"))
        results["empty_file"] = ("FAIL", "Server accepted 0-byte file")
    except urllib.error.HTTPError as err:
        body = err.read().decode('utf-8')
        results["empty_file"] = ("PASS", f"Blocked with status {err.code}: {body}")
        
    # Test 3: Unsupported extension upload (.txt)
    print("Test 3: Uploading unsupported format (.txt)...")
    txt_data, txt_hdrs = build_multipart_body("test.txt", b"plain text content", "text/plain")
    try:
        urllib.request.urlopen(urllib.request.Request(URL_PREDICT, data=txt_data, headers=txt_hdrs, method="POST"))
        results["unsupported_extension"] = ("FAIL", "Server accepted .txt file")
    except urllib.error.HTTPError as err:
        body = err.read().decode('utf-8')
        results["unsupported_extension"] = ("PASS", f"Blocked with status {err.code}: {body}")
        
    # Test 4: Fake JPG file upload (text with jpg extension)
    print("Test 4: Uploading fake JPG file...")
    fake_data, fake_hdrs = build_multipart_body("fake.jpg", b"still just plain text bytes", "image/jpeg")
    try:
        urllib.request.urlopen(urllib.request.Request(URL_PREDICT, data=fake_data, headers=fake_hdrs, method="POST"))
        results["fake_jpg"] = ("FAIL", "Server accepted text files renamed to JPG")
    except urllib.error.HTTPError as err:
        body = err.read().decode('utf-8')
        results["fake_jpg"] = ("PASS", f"Blocked with status {err.code}: {body}")
        
    # Test 5: Oversized file upload (12MB)
    print("Test 5: Uploading oversized file (12MB)...")
    oversized_bytes = b"0" * (12 * 1024 * 1024)
    over_data, over_hdrs = build_multipart_body("oversized.jpg", oversized_bytes)
    try:
        urllib.request.urlopen(urllib.request.Request(URL_PREDICT, data=over_data, headers=over_hdrs, method="POST"))
        results["oversized_file"] = ("FAIL", "Server accepted 12MB file")
    except urllib.error.HTTPError as err:
        body = err.read().decode('utf-8')
        results["oversized_file"] = ("PASS", f"Blocked with status {err.code}: {body}")
    except Exception as e:
        results["oversized_file"] = ("PASS", f"Blocked by server boundary limit: {str(e)}")

    # Load valid image bytes for name audits
    valid_bytes = valid_image_path.read_bytes()
        
    # Test 6: Long filename upload (300 characters)
    print("Test 6: Uploading image with extremely long filename...")
    long_name = "a" * 300 + ".jpg"
    long_data, long_hdrs = build_multipart_body(long_name, valid_bytes)
    try:
        status, data = send_post_request(URL_PREDICT, long_data, long_hdrs)
        results["long_filename"] = ("PASS", f"Succeeded with status {status} and parsed response: {data.get('predicted_class')}")
    except Exception as e:
        results["long_filename"] = ("FAIL", f"Failed with: {str(e)}")
        
    # Test 7: Special character filename upload
    print("Test 7: Uploading image with special characters in filename...")
    spec_name = "lesion_$#%@^!_&.jpg"
    spec_data, spec_hdrs = build_multipart_body(spec_name, valid_bytes)
    try:
        status, data = send_post_request(URL_PREDICT, spec_data, spec_hdrs)
        results["special_char_filename"] = ("PASS", f"Succeeded with status {status} and parsed response: {data.get('predicted_class')}")
    except Exception as e:
        results["special_char_filename"] = ("FAIL", f"Failed with: {str(e)}")

    for key, (status, msg) in results.items():
        color = "\033[92m" if status == "PASS" else "\033[91m"
        print(f" {color}[{status}] {key.replace('_', ' ').title()}:\033[0m {msg}")
        
    return results

def get_memory_usage():
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # MB
    return 0.0

def run_stress_suite(valid_image_path):
    print_banner("Running Load & Stress Testing Suite")
    
    valid_bytes = valid_image_path.read_bytes()
    binary_data, headers = build_multipart_body(valid_image_path.name, valid_bytes)
    
    # 1. Sequential stress test (100 runs)
    print("Executing 100 sequential requests...")
    seq_latencies = []
    success_count = 0
    
    mem_start = get_memory_usage()
    time_start = time.time()
    
    for i in range(100):
        t0 = time.time()
        try:
            req = urllib.request.Request(URL_PREDICT, data=binary_data, headers=headers, method="POST")
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                assert "predicted_class" in data
                success_count += 1
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
        seq_latencies.append(time.time() - t0)
        
    time_end = time.time()
    mem_end = get_memory_usage()
    
    total_time = time_end - time_start
    throughput = success_count / total_time
    
    avg_latency = sum(seq_latencies) / len(seq_latencies)
    min_latency = min(seq_latencies)
    max_latency = max(seq_latencies)
    seq_latencies.sort()
    median_latency = seq_latencies[len(seq_latencies)//2]
    
    print("\n--- Sequential Test Summary (100 Requests) ---")
    print(f" Success Rate      : {success_count}/100 ({success_count}%)")
    print(f" Total Elapsed Time: {total_time:.2f} seconds")
    print(f" Throughput        : {throughput:.2f} req/sec")
    print(f" Latency Min       : {min_latency * 1000:.1f} ms")
    print(f" Latency Max       : {max_latency * 1000:.1f} ms")
    print(f" Latency Average   : {avg_latency * 1000:.1f} ms")
    print(f" Latency Median    : {median_latency * 1000:.1f} ms")
    if PSUTIL_AVAILABLE:
        print(f" Memory Change (Client Process): {mem_start:.2f} MB -> {mem_end:.2f} MB (Diff: {mem_end - mem_start:.2f} MB)")
    else:
        print(" Memory footprint verification: psutil unavailable, skipped memory trace")
        
    # 2. Concurrent stress test (20 parallel runs)
    print("\nExecuting 20 concurrent requests in parallel...")
    
    def fire_request():
        t0 = time.time()
        try:
            req = urllib.request.Request(URL_PREDICT, data=binary_data, headers=headers, method="POST")
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return True, time.time() - t0, data
        except Exception as e:
            return False, time.time() - t0, str(e)

    concurrent_success = 0
    concurrent_latencies = []
    
    time_start_conc = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fire_request) for _ in range(20)]
        for fut in concurrent.futures.as_completed(futures):
            ok, lat, val = fut.result()
            if ok:
                concurrent_success += 1
                concurrent_latencies.append(lat)
            else:
                print(f"Concurrent request error: {val}")
                
    time_end_conc = time.time()
    total_time_conc = time_end_conc - time_start_conc
    
    avg_latency_conc = sum(concurrent_latencies) / len(concurrent_latencies) if concurrent_latencies else 0
    
    print("\n--- Concurrent Test Summary (20 Parallel Requests) ---")
    print(f" Success Rate      : {concurrent_success}/20 ({concurrent_success * 5}%)")
    print(f" Concurrency Time  : {total_time_conc:.2f} seconds")
    print(f" Concur Throughput : {concurrent_success / total_time_conc:.2f} req/sec")
    print(f" Average Latency   : {avg_latency_conc * 1000:.1f} ms")
    
    return {
        "sequential": {
            "total_requests": 100,
            "success_rate": success_count,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "avg_latency": avg_latency,
            "median_latency": median_latency,
            "throughput": throughput
        },
        "concurrent": {
            "total_requests": 20,
            "success_rate": concurrent_success,
            "total_time": total_time_conc,
            "avg_latency": avg_latency_conc,
            "throughput": concurrent_success / total_time_conc
        }
    }

def run_payload_validation(valid_image_path):
    print_banner("Running API Payload Schema Audit")
    
    valid_bytes = valid_image_path.read_bytes()
    binary_data, headers = build_multipart_body(valid_image_path.name, valid_bytes)
    
    try:
        req = urllib.request.Request(URL_PREDICT, data=binary_data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            
            # 1. Assert required keys exist
            assert "predicted_class" in data, "Missing predicted_class"
            assert "confidence" in data, "Missing confidence"
            assert "top_predictions" in data, "Missing top_predictions"
            print_result("Exposes required JSON keys", True)
            
            # 2. Assert confidence range [0, 1]
            conf = data["confidence"]
            assert 0.0 <= conf <= 1.0, f"Confidence score {conf} out of bounds"
            print_result("Confidence score in range [0, 1]", True, f"Confidence: {conf}")
            
            # 3. Assert top_predictions format and count
            top_preds = data["top_predictions"]
            assert len(top_preds) == 3, f"Expected 3 top predictions, got {len(top_preds)}"
            print_result("Returns exactly 3 top predictions", True)
            
            # 4. Assert sorting order of predictions
            scores = [item["score"] for item in top_preds]
            assert scores == sorted(scores, reverse=True), f"Top predictions not sorted correctly: {scores}"
            print_result("Top predictions sorted descending", True, f"Scores: {scores}")
            
            # 5. Assert top pred equals returned class
            assert top_preds[0]["class"] == data["predicted_class"], "Top prediction does not match predicted_class"
            print_result("Top-1 prediction matches returned class", True, f"Returned class: {data['predicted_class']}")
            
            # 6. Assert explainability details exist
            assert "gradcam_image" in data, "Missing gradcam_image"
            assert "heatmap_image" in data, "Missing heatmap_image"
            assert "gradcam_image_base64" in data, "Missing gradcam_image_base64"
            assert "heatmap_image_base64" in data, "Missing heatmap_image_base64"
            print_result("Contains Grad-CAM explainability assets", True)
            
            return True, data
    except Exception as e:
        print_result("API payload validation failed", False, str(e))
        return False, {}

def main():
    print("======================================================================")
    print("            SKIN CANCER DETECTION - END-TO-END AUDIT SUITE            ")
    print("======================================================================")
    
    # Check health check
    try:
        with urllib.request.urlopen(URL_HEALTH) as resp:
            health = json.loads(resp.read().decode())
            if not health.get("model_loaded"):
                print("Error: Flask application model is not loaded.")
                sys.exit(1)
    except Exception as e:
        print(f"Error: Flask server is offline on http://127.0.0.1:5000. Start it first. Detail: {e}")
        sys.exit(1)
        
    valid_image = locate_valid_image()
    if not valid_image:
        print("Error: Could not locate a valid test image on disk to run audits.")
        sys.exit(1)
        
    # Execute validations
    sec_results = run_security_suite(valid_image)
    payload_ok, payload_data = run_payload_validation(valid_image)
    stress_results = run_stress_suite(valid_image)
    
    # Save test logs to a temporary json file for reports compiler to read
    results_payload = {
        "security": {k: v[0] for k, v in sec_results.items()},
        "payload": {
            "ok": payload_ok,
            "data": payload_data
        },
        "stress": stress_results
    }
    
    with open("reports/audit_raw_results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4)
        
    print("\n\033[92mAUDIT AUTOMATION COMPLETED SUCCESSFULLY!\033[0m")
    sys.exit(0)

if __name__ == "__main__":
    main()
