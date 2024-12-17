import logging
import os

# Define the logger configuration
def setup_logger(log_dir='logs', log_file='ml_project.log', level=logging.INFO):
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_file)

    # Configure logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),  # Log to file
            logging.StreamHandler()          # Log to console
        ]
    )

    # Return the logger instance
    logger = logging.getLogger()

    return logger

# Example usage
if __name__ == '__main__':
    logger = setup_logger()

    logger.info("Logger is set up successfully!")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning.")
    logger.error("This is an error.")
    logger.critical("This is a critical error.")
