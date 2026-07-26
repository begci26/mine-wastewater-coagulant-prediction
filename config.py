import os

class Config:
    """Flask application configuration class."""
    # Security key for session signing
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'development_secret_key_coagulant_prediction_9921'
    
    # Base directory of the application
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Upload configurations
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads', 'dataset')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB limit
    ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
    
    # Model saving directory
    MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
    
    # Output directories
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
    
    # Ensure directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    
    # Ensure all output subfolders exist
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'preprocessing'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'eda'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'training'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'evaluasi'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'prediksi'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'bab4'), exist_ok=True)

