import logging
import os

def setup_logging(log_dir="logs", log_file="training.log"):
    """
    Sets up the logging configuration for the project.
    Logs will be saved to both console and a log file.
    
    Args:
        log_dir (str): Directory where log file will be saved.
        log_file (str): Name of the log file.
    """
    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Define log file path
    log_path = os.path.join(log_dir, log_file)

    # Create a custom logger
    logger = logging.getLogger()

    # Set the default logging level
    logger.setLevel(logging.DEBUG)

    # Create a console handler for output to the terminal
    console_handler = logging.StreamHandler()

    # Create a file handler to save logs to a file
    file_handler = logging.FileHandler(log_path)

    # Define a common log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Attach the formatter to both handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log the start of the training or process
    logger.info("Logging setup complete.")
    logger.info(f"Logs will be saved to {log_path}")

    return logger

def log_metrics(logger, metrics, epoch, mode='train'):
    """
    Log metrics like loss, accuracy, etc., during training or validation.
    
    Args:
        logger (logging.Logger): Logger object.
        metrics (dict): Metrics to log (e.g., loss, accuracy).
        epoch (int): Current epoch number.
        mode (str): Either 'train' or 'val' to specify the mode of logging.
    """
    logger.info(f"Epoch {epoch} - {mode} metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")

def log_error(logger, error_message):
    """
    Log an error message to both console and file.
    
    Args:
        logger (logging.Logger): Logger object.
        error_message (str): The error message to log.
    """
    logger.error(f"Error: {error_message}")

def log_warning(logger, warning_message):
    """
    Log a warning message to both console and file.
    
    Args:
        logger (logging.Logger): Logger object.
        warning_message (str): The warning message to log.
    """
    logger.warning(f"Warning: {warning_message}")
