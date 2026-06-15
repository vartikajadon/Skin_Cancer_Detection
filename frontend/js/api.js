// API Service Layer for Skin Cancer Detection App
(function() {
  const getApiUrl = (endpoint) => {
    const baseUrl = (window.CONFIG && window.CONFIG.API_BASE_URL) || 'http://127.0.0.1:5000';
    return `${baseUrl}${endpoint}`;
  };

  const apiService = {
    /**
     * Checks backend server health and model load status.
     * @returns {Promise<Object>} Status object.
     */
    async healthCheck() {
      try {
        const response = await fetch(getApiUrl('/api/health'), {
          method: 'GET',
          headers: { 'Accept': 'application/json' }
        });
        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.message || errData.error || `Server health check failed with status: ${response.status}`);
        }
        return await response.json();
      } catch (err) {
        console.error('Health Check API failure:', err);
        throw err;
      }
    },

    /**
     * Sends skin lesion image file for prediction.
     * @param {File} file The image file.
     * @returns {Promise<Object>} Prediction response data.
     */
    async predictLesion(file) {
      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(getApiUrl('/api/predict'), {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.message || errData.error || `Prediction request failed with status: ${response.status}`);
        }

        return await response.json();
      } catch (err) {
        // Handle network drops or API unavailability
        if (err.name === 'TypeError' && err.message.includes('fetch')) {
          throw new Error('Backend server is unreachable. Please ensure the Flask API service is running.');
        }
        throw err;
      }
    }
  };

  // Expose to global window scope
  window.apiService = apiService;
})();

