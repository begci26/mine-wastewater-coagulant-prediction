import os
from flask import Flask, render_template, redirect, url_for
from config import Config
from utils.logger import app_logger

def create_app():
    """Application factory method to configure and initialize Flask."""
    app = Flask(__name__)
    app.config.from_object(Config)

    @app.template_filter("metric4")
    def format_metric_for_ui(value):
        """Format a metric for compact UI display without mutating its stored precision."""
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "N/A"
    
    # Register Blueprints
    from routes.dashboard import dashboard_bp
    from routes.dataset import dataset_bp
    from routes.preprocessing import preprocessing_bp
    from routes.eda import eda_bp
    from routes.training import training_bp
    from routes.evaluation import evaluation_bp
    from routes.prediction import prediction_bp
    from routes.about import about_bp
    
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(dataset_bp, url_prefix='/dataset')
    app.register_blueprint(preprocessing_bp, url_prefix='/preprocessing')
    app.register_blueprint(eda_bp, url_prefix='/eda')
    app.register_blueprint(training_bp, url_prefix='/training')
    app.register_blueprint(evaluation_bp, url_prefix='/evaluation')
    app.register_blueprint(prediction_bp, url_prefix='/prediction')
    app.register_blueprint(about_bp, url_prefix='/about')
    
    # Auto Session Recovery from disk files
    @app.before_request
    def auto_restore_session():
        from flask import session
        import json
        from utils.helpers import check_dataset_uploaded
        
        # 1. Dataset Uploaded
        if not session.get('dataset_uploaded') and check_dataset_uploaded():
            session['dataset_uploaded'] = True
            
        # 2. Preprocessing Complete
        if not session.get('preprocessing_complete'):
            processed_dir = os.path.join(Config.UPLOAD_FOLDER, 'processed')
            state_path = os.path.join(processed_dir, 'preprocessing_state.json')
            if os.path.exists(state_path):
                try:
                    with open(state_path, 'r') as f:
                        state = json.load(f)
                    if state.get('split'):
                        session['preprocessing_complete'] = True
                        session['preprocessing_results'] = {
                            'imputation': state.get('missing_strategy', 'drop'),
                            'scaler': state.get('scaler_strategy', 'standard'),
                            'test_size_percentage': state.get('split_ratio', 20.0),
                            'train_rows': state.get('train_shape', [0, 0])[0],
                            'train_cols': state.get('train_shape', [0, 11])[1],
                            'test_rows': state.get('test_shape', [0, 0])[0],
                            'test_cols': state.get('test_shape', [0, 11])[1]
                        }
                except Exception as error:
                    app_logger.error(f"Error restoring preprocessing session: {error}")
                    
        # 3. Training & Evaluation Complete
        if not session.get('training_complete') or not session.get('evaluation_complete'):
            eval_results_path = os.path.join(Config.MODEL_FOLDER, 'evaluation_results.joblib')
            metadata_path = os.path.join(Config.MODEL_FOLDER, 'model_metadata.json')
            if os.path.exists(eval_results_path) and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    from utils.helpers import ARTIFACT_SCHEMA_VERSION, MODEL_FEATURES
                    if (
                        metadata.get('artifact_schema_version') != ARTIFACT_SCHEMA_VERSION
                        or metadata.get('feature_names') != MODEL_FEATURES
                    ):
                        raise ValueError("Artefak model lama tidak kompatibel dengan skema 11 fitur.")
                    session['training_complete'] = True
                    session['evaluation_complete'] = True
                    session['best_model_name'] = metadata.get('best_model')
                    session['best_model_rmse'] = metadata.get('best_metrics', {}).get('rmse')
                except Exception as error:
                    app_logger.error(f"Error restoring training session: {error}")

    # Root route redirection
    @app.route('/')
    def index():
        return redirect(url_for('dashboard.index'))
        
    # Error Handlers
    @app.errorhandler(404)
    def page_not_found(error):
        app_logger.warning(f"404 Error: {error}")
        return render_template('error.html', error_code=404, 
                               error_title="Page Not Found", 
                               error_message="The page you are looking for does not exist or has been moved."), 404
                               
    @app.errorhandler(500)
    def internal_server_error(error):
        app_logger.error(f"500 Error: {error}", exc_info=True)
        return render_template('error.html', error_code=500, 
                               error_title="Internal Server Error", 
                               error_message="An unexpected error occurred on the server. Please check the logs."), 500
                               
    @app.errorhandler(413)
    def file_too_large(error):
        app_logger.warning(f"413 Error: {error}")
        return render_template('error.html', error_code=413,
                               error_title="File Too Large",
                               error_message="The uploaded file exceeds the maximum allowed size limit of 10MB."), 413
                               
    return app

if __name__ == '__main__':
    app = create_app()
    app_logger.info("Starting Coagulant Prediction Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
