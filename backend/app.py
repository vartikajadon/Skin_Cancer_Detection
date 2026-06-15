import os
import sys
import logging
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS

# Configure path variables to import modules from backend/ and src/
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
root_dir = backend_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from routes import api_bp
from services.prediction_service import initialize_prediction_service

def create_app(
    model_path: str = "models/efficientnet_b0_best.keras",
    encoder_path: str = "processed/label_encoder.json"
) -> Flask:
    """Flask Application Factory."""
    # Serve frontend assets directly from the frontend directory
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
    
    # 1. Enable CORS for secure cross-origin API client interaction
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # 2. Register Routes Blueprint
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Serve frontend index.html from root route
    @app.route('/')
    def serve_frontend_root():
        return app.send_static_file('index.html')
    
    # 3. Configure Upload parameters
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # 10MB limit
    
    # Create folders if missing
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # 4. Initialize prediction service singleton once at server start
    try:
        initialize_prediction_service(Path(model_path), Path(encoder_path))
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load classification model during startup: {str(e)}")
        # In a real environment we might choose to crash, but for robustness
        # we let app start and routes will return 500 when calling uninitialized model.
        
    # Serve static files from gradcam_outputs directory
    gradcam_outputs_dir = os.path.join(os.getcwd(), 'gradcam_outputs')
    os.makedirs(gradcam_outputs_dir, exist_ok=True)
    
    @app.route('/gradcam_outputs/<path:filename>')
    def serve_gradcam_output(filename):
        from flask import send_from_directory
        return send_from_directory(gradcam_outputs_dir, filename)
        
    # 5. Register Generic Global Error Handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "Bad Request: Invalid payload or parameter structure"}), 400
        
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Resource Not Found"}), 404
        
    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({"error": "Internal Server Error: Predictor pipeline failed"}), 500
        
    return app

if __name__ == "__main__":
    app = create_app()
    # Run server locally on port 5000
    logger.info("Starting local Flask server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
