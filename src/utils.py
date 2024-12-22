import os
import pickle
from src.logger import setup_logger
from src.exception import CustomException

# Logger setup
logger = setup_logger()

def save_pickle_file(obj, file_path):
    """
    Saves a Python object as a pickle file.

    Args:
        obj: The Python object to save.
        file_path: The full file path where the pickle file will be saved.

    Raises:
        CustomException: If there is an error during the saving process.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the object as a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        
        logger.info(f"Pickle file saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error saving pickle file: {e}")
        raise CustomException(f"Error saving pickle file: {e}")
