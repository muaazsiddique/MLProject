import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import setup_logger
from src.exception import DataLoadException, CustomException

# Logger setup
logger = setup_logger()

class DataIngestionConfig:
    """Configuration for data ingestion."""
    data_file_path = r"E:\Projects\MLProject\notebook\data\StudentsPerformance.csv"
    output_dir = r"E:\Projects\MLProject\artifacts"  # Save in artifacts folder
    raw_file_name = "raw_data.csv"  # File name for raw data
    train_file_name = "train.csv"
    test_file_name = "test.csv"
    test_size = 0.2  # Proportion of data to use for testing
    random_state = 42  # For reproducibility of train-test split

class DataIngestion:
    """Class to handle data ingestion."""
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def read_data(self):
        """Reads data from the specified file path."""
        try:
            logger.info(f"Attempting to read data from {self.config.data_file_path}")
            
            # Check if file exists
            if not os.path.exists(self.config.data_file_path):
                raise DataLoadException(f"Data file not found at {self.config.data_file_path}")
            
            # Load the dataset
            data = pd.read_csv(self.config.data_file_path)
            logger.info("Data loaded successfully.")
            logger.info(f"Data shape: {data.shape}")
            
            return data
        except Exception as e:
            logger.error(f"Error during data loading: {e}")
            raise DataLoadException(f"Error during data loading: {e}")
    
    def save_raw_data(self, data):
        """Saves the raw data to the specified output directory."""
        try:
            logger.info(f"Saving raw data to {self.config.output_dir}")
            
            # Create output directory if it doesn't exist
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Save raw data
            raw_file_path = os.path.join(self.config.output_dir, self.config.raw_file_name)
            data.to_csv(raw_file_path, index=False)
            
            logger.info(f"Raw data saved to {raw_file_path}")
        except Exception as e:
            logger.error(f"Error during saving raw data: {e}")
            raise CustomException(f"Error during saving raw data: {e}")
    
    def split_data(self, data):
        """Splits the data into training and testing datasets."""
        try:
            logger.info("Splitting data into training and testing sets.")
            
            # Perform train-test split
            train_data, test_data = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            logger.info(f"Training data shape: {train_data.shape}")
            logger.info(f"Testing data shape: {test_data.shape}")
            
            return train_data, test_data
        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise CustomException(f"Error during data splitting: {e}")
    
    def save_data(self, train_data, test_data):
        """Saves training and testing datasets to the specified output directory."""
        try:
            logger.info(f"Saving training and testing datasets to {self.config.output_dir}")
            
            # Save training and testing datasets
            train_file_path = os.path.join(self.config.output_dir, self.config.train_file_name)
            test_file_path = os.path.join(self.config.output_dir, self.config.test_file_name)
            
            train_data.to_csv(train_file_path, index=False)
            test_data.to_csv(test_file_path, index=False)
            
            logger.info(f"Training data saved to {train_file_path}")
            logger.info(f"Testing data saved to {test_file_path}")
        except Exception as e:
            logger.error(f"Error during data saving: {e}")
            raise CustomException(f"Error during data saving: {e}")
    
    def execute(self):
        """Executes the data ingestion process."""
        try:
            # Step 1: Read the data
            data = self.read_data()
            
            # Step 2: Save raw data
            self.save_raw_data(data)
            
            # Step 3: Split the data
            train_data, test_data = self.split_data(data)
            
            # Step 4: Save the split data
            self.save_data(train_data, test_data)
            
            logger.info("Data ingestion process completed successfully.")
            return data, train_data, test_data
        except Exception as e:
            logger.critical(f"Data ingestion process failed: {e}")
            raise CustomException(f"Data ingestion process failed: {e}")

if __name__ == "__main__":
    # Instantiate the configuration and ingestion class
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    
    # Execute the data ingestion process
    try:
        raw_data, train_data, test_data = data_ingestion.execute()
    except CustomException as e:
        logger.critical(f"Data ingestion failed with error: {e}")
