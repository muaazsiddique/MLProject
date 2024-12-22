import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.logger import setup_logger
from src.exception import CustomException, DataPreprocessingException
from src.utils import save_pickle_file  # Import the utility function


# Logger setup
logger = setup_logger()

class DataTransformationConfig:
    """Configuration for data transformation."""
    output_dir = r"E:\Projects\MLProject\artifacts"
    train_features_file_name = "train_features.csv"
    test_features_file_name = "test_features.csv"
    train_target_file_name = "train_target.csv"
    test_target_file_name = "test_target.csv"
    transformer_file_name = "preprocessor.pkl"

class DataTransformation:
    """Class to handle data transformation."""
    
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def get_preprocessor(self, data):
        """Creates a preprocessing pipeline."""
        try:
            logger.info("Creating preprocessing pipeline.")

            # Separate numerical and categorical columns (excluding target)
            numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numerical_features.remove('math score')  # Exclude target column
            categorical_features = data.select_dtypes(include=['object']).columns.tolist()
            
            logger.info(f"Numerical features: {numerical_features}")
            logger.info(f"Categorical features: {categorical_features}")

            # Define numerical pipeline
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Define categorical pipeline
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine pipelines using ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ])

            logger.info("Preprocessing pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logger.error(f"Error during pipeline creation: {e}")
            raise DataPreprocessingException(f"Error during pipeline creation: {e}")

    def apply_transformation(self, train_data, test_data, preprocessor):
        """Applies the preprocessing pipeline to training and testing datasets."""
        try:
            logger.info("Applying transformations to training and testing datasets.")

            # Separate features and target
            X_train = train_data.drop(columns=['math score'])
            y_train = train_data['math score']
            X_test = test_data.drop(columns=['math score'])
            y_test = test_data['math score']

            # Fit the preprocessor on training data
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logger.info("Transformations applied successfully.")
            return X_train_transformed, X_test_transformed, y_train, y_test

        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise DataPreprocessingException(f"Error during data transformation: {e}")

    def save_preprocessor(self, preprocessor):
        """Saves the preprocessing pipeline using the utility function."""
        try:
            logger.info(f"Saving preprocessing pipeline to {self.config.output_dir}")
            
            pickle_path = os.path.join(self.config.output_dir, self.config.transformer_file_name)
            
            # Use the utility function to save the preprocessor
            save_pickle_file(preprocessor, pickle_path)
        
        except Exception as e:
            logger.error(f"Error saving preprocessor: {e}")
            raise CustomException(f"Error saving preprocessor: {e}")

    def save_transformed_data(self, X_train, X_test, y_train, y_test):
        """Saves the transformed data and targets to the specified output directory."""
        try:
            logger.info(f"Saving transformed data and targets to {self.config.output_dir}")

            # Save features
            train_features_path = os.path.join(self.config.output_dir, self.config.train_features_file_name)
            test_features_path = os.path.join(self.config.output_dir, self.config.test_features_file_name)
            pd.DataFrame(X_train).to_csv(train_features_path, index=False)
            pd.DataFrame(X_test).to_csv(test_features_path, index=False)

            # Save targets
            train_target_path = os.path.join(self.config.output_dir, self.config.train_target_file_name)
            test_target_path = os.path.join(self.config.output_dir, self.config.test_target_file_name)
            y_train.to_csv(train_target_path, index=False, header=True)
            y_test.to_csv(test_target_path, index=False, header=True)

            logger.info(f"Transformed features saved to {train_features_path} and {test_features_path}")
            logger.info(f"Targets saved to {train_target_path} and {test_target_path}")

        except Exception as e:
            logger.error(f"Error during saving transformed data: {e}")
            raise CustomException(f"Error during saving transformed data: {e}")

    def execute(self, train_data, test_data):
        """Executes the data transformation process."""
        try:
            # Step 1: Get the preprocessing pipeline
            preprocessor = self.get_preprocessor(train_data)

            # Step 2: Apply transformation
            X_train, X_test, y_train, y_test = self.apply_transformation(train_data, test_data, preprocessor)

            # Step 3: Save transformed data
            self.save_transformed_data(X_train, X_test, y_train, y_test)

            # Step 4: Save the preprocessor
            self.save_preprocessor(preprocessor)

            logger.info("Data transformation process completed successfully.")
            return X_train, X_test, y_train, y_test

        except CustomException as e:
            logger.critical(f"Data transformation process failed with error: {e}")
            raise e

        except Exception as e:
            logger.critical(f"Unexpected error in data transformation process: {e}")
            raise CustomException(f"Unexpected error in data transformation process: {e}")

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig


    # Step 1: Data ingestion
    ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion(ingestion_config)
    raw_data, train_data, test_data = data_ingestion.execute()

    # Step 2: Data transformation
    transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(transformation_config)

    try:
        X_train, X_test, y_train, y_test = data_transformation.execute(train_data, test_data)
    except CustomException as e:
        logger.critical(f"Data transformation failed with error: {e}")
