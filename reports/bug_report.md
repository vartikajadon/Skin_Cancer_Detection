# Skin Cancer Detection System - Bug Audit Report

This report catalogs all identified bugs, edge cases, vulnerabilities, and code quality weaknesses discovered during the end-to-end audit.

---

## 1. Summary of Discovered Bugs & Issues

| ID | Issue Title | Component | Severity | Status |
| :--- | :--- | :--- | :--- | :--- |
| **BUG-01** | Windows `MAX_PATH` File-Save Vulnerability | Backend API | **CRITICAL** | **OPEN** |
| **BUG-02** | Missing Frontend Filename Length Restriction | Frontend UI | **MEDIUM** | **OPEN** |
| **BUG-03** | Lack of Automated Temporary Directory Housekeeping | Backend Utils | **LOW** | **OPEN** |
| **BUG-04** | Overly Permissive CORS Configuration | Backend Config | **LOW** | **OPEN** |

---

## 2. Detailed Bug Reports

### BUG-01: Windows `MAX_PATH` File-Save Vulnerability
* **Component**: [routes.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/backend/routes.py#L41-L49) (L41-49)
* **Severity**: **CRITICAL** (Security and Routing Vulnerability)
* **Impact**: Bypasses standard validation logic. Windows filesystem paths are limited to **260 characters** (`MAX_PATH`) by default. When an upload has an extremely long filename (e.g. >250 characters), constructing `temp_path = UPLOAD_DIR / unique_filename` creates a path of ~270+ characters. Calling `file.save(str(temp_path))` fails at the OS level, raising a `FileNotFoundError`. The backend catches this specific exception and returns a **404 Resource Not Found** JSON error instead of a **400 Bad Request** or gracefully sanitizing/truncating the file.
* **Steps to Reproduce**:
  1. Start the Flask backend server.
  2. Send a POST request to `/api/predict` with a file payload whose name contains 300 characters.
  3. The server crashes inside the `try` block before validations run and responds with a `404 Not Found` response.
* **Code Snippet**:
  ```python
  filename = secure_filename(file.filename)
  # Append timestamp to prevent collision during concurrent tests
  import time
  unique_filename = f"{int(time.time() * 1000)}_{filename}"
  temp_path = UPLOAD_DIR / unique_filename
  
  try:
      file.save(str(temp_path))  # <-- Throws FileNotFoundError if path > 260 chars on Windows
  ```
* **Recommended Fix**:
  Truncate the parsed `secure_filename` to a maximum length (e.g. 50 characters) before appending the unique timestamp:
  ```python
  base_name, ext = os.path.splitext(secure_filename(file.filename))
  truncated_name = base_name[:50]
  unique_filename = f"{int(time.time() * 1000)}_{truncated_name}{ext}"
  ```

---

### BUG-02: Missing Frontend Filename Length Restriction
* **Component**: [app.js](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/js/app.js#L156-L196) (L156-196)
* **Severity**: **MEDIUM** (Input Validation Defect)
* **Impact**: The frontend validation layer in `handleFileSelection` properly checks the file format and file size limits (5MB), but it fails to sanitize or limit the length of the filename string. This allows invalid or overly long filename strings to be passed directly to the fetch API payload, leading to the backend crash detailed in **BUG-01**.
* **Steps to Reproduce**:
  1. Open the UI in a browser.
  2. Drop an image file named `a` * 300 + `.jpg`.
  3. The frontend accepts the file and fires the API request, which then returns a raw error response in the UI drop-zone.
* **Recommended Fix**:
  Add an explicit filename length check inside the frontend selection handler:
  ```javascript
  if (file.name.length > 100) {
    showErrorCard('Filename is too long. Please rename the file to under 100 characters before uploading.');
    return;
  }
  ```

---

### BUG-03: Lack of Automated Temporary Directory Housekeeping
* **Component**: [routes.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/backend/routes.py) & [app.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/backend/app.py)
* **Severity**: **LOW** (Operational Maintenance Risk)
* **Impact**: While the predict route attempts to delete temporary files immediately after inference inside the `try`, `except ImageValidationError`, and `except Exception` blocks, a sudden hardware crash, process termination (`kill -9`), or system reboot while processing a request will leave abandoned files inside the `uploads/` directory indefinitely. Over time, this can lead to disk space exhaustion.
* **Steps to Reproduce**:
  1. Upload multiple images for prediction.
  2. Force-kill the Flask process immediately after files are written to `uploads/` but before cleanup.
  3. Mapped files will persist on disk in the `uploads/` folder.
* **Recommended Fix**:
  Initialize a startup check or a background cron thread in `app.py` to clear the `uploads/` directory on server launch or at regular intervals:
  ```python
  # Clear existing files on app startup
  import shutil
  if os.path.exists(app.config['UPLOAD_FOLDER']):
      for filename in os.listdir(app.config['UPLOAD_FOLDER']):
          file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
          try:
              if os.path.isfile(file_path) or os.path.islink(file_path):
                  os.unlink(file_path)
          except Exception as e:
              logger.warning(f"Failed to clear startup temp file {file_path}: {e}")
  ```

---

### BUG-04: Overly Permissive CORS Configuration
* **Component**: [app.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/backend/app.py#L35-L36) (L35-36)
* **Severity**: **LOW** (Security/Access Control Weakness)
* **Impact**: The CORS configuration uses a wildcard origin `origins: "*"` for all `/api/*` endpoints. While appropriate for local debug and testing, this leaves the API vulnerable to unauthorized cross-origin requests from malicious websites if deployed without an explicit whitelist.
* **Recommended Fix**:
  Make CORS origins configurable via environment variables, defaulting to specific domain configurations in production:
  ```python
  allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
  CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
  ```
