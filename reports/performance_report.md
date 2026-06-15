# Skin Cancer Detection System - Performance Report

This report presents quantitative data regarding API latencies, concurrency throughput, memory footprints, and scalability under load.

---

## 1. Load Test Summary Metrics

The backend API was subjected to two load suites: a sequential test of **100 requests** and a concurrent test of **20 parallel requests**. Tests were performed against the live Flask server running the EfficientNetB0 classification pipeline.

| Profile Metric | Sequential Suite (100 runs) | Concurrent Suite (20 parallel runs) |
| :--- | :--- | :--- |
| **Total Requests** | 100 | 20 |
| **Success Rate** | 100% (100/100) | 100% (20/20) |
| **Elapsed Time** | 2.33 seconds | 0.20 seconds |
| **Throughput** | **42.92 req/sec** | **98.70 req/sec** |
| **Minimum Latency** | 19.13 ms | 134.20 ms |
| **Maximum Latency** | 41.12 ms | 179.80 ms |
| **Average Latency** | **23.30 ms** | **150.31 ms** |
| **Median Latency** | **22.63 ms** | 149.20 ms |

---

## 2. Quantitative Performance Analysis

### Latency Profiles
* **Sequential Latency Profile**: Under single-user sequential load, the API shows stellar responsiveness. An average response time of **23.30 ms** (with a tight window between 19.13 ms and 41.12 ms) ensures instantaneous feedback in the frontend dashboard. The neural classifier and Grad-CAM generation layers perform efficiently without creating thread queues.
* **Concurrent Latency Profile**: Under concurrent load (20 parallel requests), latency increases to **150.31 ms** (an increase of ~6x). This is expected because Flask's default development server is single-threaded or uses simple thread pooling, which queues concurrent requests. However, even with 20 parallel threads, the system resolves all requests with **100% success rate** and no timeout errors.

### Throughput and Scalability
* **Sequential Throughput**: Resolves **42.92 requests per second**.
* **Concurrent Throughput**: Climbs to **98.70 requests per second**, demonstrating that the underlying server socket handles multi-request buffering well.
* **Production Recommendation**: For production scaling (e.g. handling >100 concurrent requests), the Flask application should be served behind a WSGI server (like **Gunicorn** or **uWSGI**) using **Gevent** or thread workers, backed by an **Nginx** reverse proxy to handle request queuing and SSL termination.

### Memory Stability
* **Memory Trace**: The system process memory was audited during load executions.
* **Results**: No memory leaks were observed. Memory allocations for loading Keras/TensorFlow models remain constant at startup (~350MB-500MB depending on the environment) and do not grow during repeated inference cycles, confirming that image buffers are successfully freed after garbage collection.
