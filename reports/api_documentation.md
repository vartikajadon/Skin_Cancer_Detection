# Skin Lesion Classification REST API Documentation (Sprint 8)

The backend provides REST API endpoints to serve skin lesion classification predictions from the trained EfficientNetB0 Transfer Learning model.

---

## Base URL
When running locally:
`http://127.0.0.1:5000`

---

## API Endpoints Reference

### 1. Health Check
Checks backend health status and confirms whether the deep learning model has successfully loaded into memory.

* **Route**: `/api/health`
* **Method**: `GET`
* **Headers**: `Content-Type: application/json`

#### Response (Success - 200 OK)
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

#### Response (Failure - 500 Internal Server Error)
```json
{
    "status": "unhealthy",
    "model_loaded": false
}
```

---

### 2. Predict Image
Accepts a skin lesion image file upload, validates image integrity, runs model inference, and returns target diagnostic classification lists.

* **Route**: `/api/predict`
* **Method**: `POST`
* **Content Type**: `multipart/form-data`
* **Request Payload**:
  * `file`: The skin lesion image binary (supported formats: `.jpg`, `.jpeg`, `.png`, max size: 10MB).

#### Response (Success - 200 OK)
```json
{
    "predicted_class": "mel",
    "confidence": 0.9348,
    "top_predictions": [
        {
            "class": "mel",
            "score": 0.9348
        },
        {
            "class": "bkl",
            "score": 0.0412
        },
        {
            "class": "nv",
            "score": 0.024
        }
    ]
}
```

#### Example cURL Command
```bash
curl -X POST -F "file=@/path/to/lesion.jpg" http://127.0.0.1:5000/api/predict
```

---

## Error Codes and Handlers

The API returns structured, user-friendly JSON error messages with appropriate HTTP status codes:

| HTTP Status | Error JSON Response | Cause / Resolution |
| :---: | :--- | :--- |
| **400 Bad Request** | `{"error": "No file part in the request"}` | Upload form key `file` was omitted from request. |
| **400 Bad Request** | `{"error": "No file selected for uploading"}` | Payload included the form key but image file was empty. |
| **400 Bad Request** | `{"error": "Unsupported file format '.pdf'..."}` | File suffix is unsupported. Only `.jpg`, `.jpeg`, and `.png` are accepted. |
| **400 Bad Request** | `{"error": "Corrupted image file..."}` | OpenCV failed to decode pixels (corrupted binary upload). |
| **400 Bad Request** | `{"error": "File size exceeds limit..."}` | Request payload exceeded maximum upload threshold (10MB). |
| **404 Not Found** | `{"error": "Resource Not Found"}` | Route endpoint does not exist. |
| **500 Internal Error** | `{"error": "Prediction Service not initialized"}` | Model failed to load at startup. Check server logs. |

---

## Local Deployment Instructions

### Prerequisites
* Python 3.9 - 3.11 (with TensorFlow and OpenCV) or Python 3.14 (with fallback mode).
* Dependencies:
  ```bash
  pip install flask flask-cors opencv-python pandas numpy
  ```

### Launching the Server
Navigate to the project root directory and execute:
```bash
python backend/app.py
```

*The Flask backend server will spin up on http://127.0.0.1:5000.*

### Automated Tests
To execute the automated API blueprint unit tests, run:
```bash
python -m unittest tests/test_api.py
```
