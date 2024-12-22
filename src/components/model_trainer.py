import os
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from src.logger import setup_logger
from src.exception import CustomException
from src.utils import save_pickle_file

# Logger setup
logger = setup_logger()

class ModelTrainerConfig:
    """Configuration for the model trainer."""
    output_dir = r"E:\Projects\MLProject\artifacts"
    best_model_file_name = "best_model.pkl"
    random_state = 42
    minimum_r2_threshold = 0.60  # Minimum R2 score threshold for the best model

class ModelTrainer:
    """Class to handle model training and evaluation."""
    
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def get_models(self):
        """Defines a dictionary of models to be trained."""
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Decision Tree": DecisionTreeRegressor(random_state=self.config.random_state),
            "Random Forest": RandomForestRegressor(random_state=self.config.random_state),
            "Gradient Boosting": GradientBoostingRegressor(random_state=self.config.random_state),
            "Support Vector Regressor": SVR(),
        }
        return models
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Trains and evaluates multiple models, and selects the best one."""
        try:
            logger.info("Starting training and evaluation of models.")
            
            models = self.get_models()
            scores = []
            best_model = None
            best_mse = float("inf")
            best_r2 = float("-inf")
            best_model_name = None

            # Iterate through models
            for model_name, model in models.items():
                logger.info(f"Training model: {model_name}")
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Evaluate the model
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                logger.info(f"Model: {model_name}, MSE: {mse}, R2: {r2}")

                # Append scores for reporting
                scores.append({"Model": model_name, "MSE": mse, "R2": r2})

                # Check if this model is better
                if mse < best_mse:
                    best_mse = mse
                    best_r2 = r2
                    best_model = model
                    best_model_name = model_name

            logger.info(f"Best model: {best_model_name}, MSE: {best_mse}, R2: {best_r2}")

            # Check if the best model meets the minimum R2 threshold
            if best_r2 < self.config.minimum_r2_threshold:
                raise CustomException(
                    f"No suitable model found. Best model R2 ({best_model_name}): {best_r2} is below the threshold of {self.config.minimum_r2_threshold}."
                )

            return best_model, best_model_name, best_mse, best_r2, scores
        except Exception as e:
            logger.error(f"Error during model training and evaluation: {e}")
            raise CustomException(f"Error during model training and evaluation: {e}")

    def save_model(self, model):
        """Saves the best model using the utility function."""
        try:
            logger.info(f"Saving the best model to {self.config.output_dir}")

            # Path to save the model
            model_path = os.path.join(self.config.output_dir, self.config.best_model_file_name)

            # Use the utility function to save the model
            save_pickle_file(model, model_path)

            logger.info(f"Best model saved successfully at {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error saving the best model: {e}")
            raise CustomException(f"Error saving the best model: {e}")

    def execute(self, X_train, X_test, y_train, y_test):
        """Executes the model training and evaluation process."""
        try:
            # Step 1: Train and evaluate multiple models
            best_model, best_model_name, best_mse, best_r2, scores = self.train_and_evaluate(X_train, X_test, y_train, y_test)

            # Step 2: Save the best model
            model_path = self.save_model(best_model)

            logger.info("Model training process completed successfully.")
            return best_model, best_model_name, best_mse, best_r2, scores, model_path
        except CustomException as e:
            logger.critical(f"Model training process failed with error: {e}")
            raise e
        except Exception as e:
            logger.critical(f"Unexpected error in model training process: {e}")
            raise CustomException(f"Unexpected error in model training process: {e}")

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig
    from src.components.data_transformation import DataTransformation, DataTransformationConfig

    # Step 1: Data ingestion
    ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion(ingestion_config)
    raw_data, train_data, test_data = data_ingestion.execute()

    # Step 2: Data transformation
    transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(transformation_config)
    X_train, X_test, y_train, y_test = data_transformation.execute(train_data, test_data)

    # Step 3: Model training
    trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(trainer_config)

    try:
        best_model, best_model_name, best_mse, best_r2, scores, model_path = model_trainer.execute(X_train, X_test, y_train, y_test)
        
        print(f"Model training completed successfully.")
        print(f"Best Model: {best_model_name}")
        print(f"Mean Squared Error (MSE): {best_mse}")
        print(f"R-Squared (R2): {best_r2}")
        print(f"Best Model saved at: {model_path}")
        print("\nAll Model Scores:")
        for score in scores:
            print(f"{score['Model']} - MSE: {score['MSE']}, R2: {score['R2']}")

    except CustomException as e:
        print(f"Model training failed: {e}")
        logger.critical(f"Model training failed with error: {e}")
