import os
import pandas as pd
import joblib
from src.exception import CustomException
from src.logger import setup_logger

# Logger setup
logger = setup_logger()

class PredictPipeline:
    """Pipeline for loading model and preprocessor and making predictions."""

    def __init__(self, model_path, preprocessor_path):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def load_artifacts(self):
        """Loads the saved model and preprocessor."""
        try:
            logger.info("Loading model and preprocessor.")
            model = joblib.load(self.model_path)
            preprocessor = joblib.load(self.preprocessor_path)
            logger.info("Model and preprocessor loaded successfully.")
            return model, preprocessor
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise CustomException(f"Error loading artifacts: {e}")

    def predict(self, input_data):
        """Processes input data and makes predictions."""
        try:
            model, preprocessor = self.load_artifacts()

            # Convert input_data to DataFrame
            input_df = pd.DataFrame(input_data)
            logger.info(f"Input data received for prediction: {input_df}")

            # Preprocess input data
            processed_data = preprocessor.transform(input_df)

            # Make predictions
            predictions = model.predict(processed_data)
            logger.info(f"Predictions made successfully: {predictions}")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise CustomException(f"Error during prediction: {e}")
