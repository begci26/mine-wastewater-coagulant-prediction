import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name="coagulant_app"):
    """Sets up and returns a custom logger instance."""
    logger = logging.getLogger(name)
    
    # If logger already has handlers, do not add more (prevents duplicate logs)
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it does not exist
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    log_path = os.path.join(logs_dir, 'app.log')
    
    # Create file handler
    file_handler = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create global logger instance
app_logger = setup_logger()
