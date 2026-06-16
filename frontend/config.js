// Global configuration for the Skin Cancer Detection frontend
const isLocal = window.location.hostname === 'localhost' || 
                window.location.hostname === '127.0.0.1' || 
                window.location.hostname === '' || 
                window.location.protocol === 'file:';

window.CONFIG = {
  API_BASE_URL: isLocal ? 'http://127.0.0.1:5000' : window.location.origin
};
