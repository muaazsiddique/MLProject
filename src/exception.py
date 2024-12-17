# exception.py

class CustomException(Exception):
    """Base class for all custom exceptions in the ML project."""
    def __init__(self, message, *args):
        self.message = message
        super().__init__(self.message, *args)

class DataLoadException(CustomException):
    """Exception raised when there is an error in loading the data."""
    def __init__(self, message="Error occurred while loading the data.", *args):
        super().__init__(message, *args)

class DataPreprocessingException(CustomException):
    """Exception raised during data preprocessing errors."""
    def __init__(self, message="Error during data preprocessing.", *args):
        super().__init__(message, *args)

class ModelTrainingException(CustomException):
    """Exception raised during model training errors."""
    def __init__(self, message="Error during model training.", *args):
        super().__init__(message, *args)

class ModelEvaluationException(CustomException):
    """Exception raised during model evaluation errors."""
    def __init__(self, message="Error during model evaluation.", *args):
        super().__init__(message, *args)

class InvalidModelInputException(CustomException):
    """Exception raised for invalid input to a model."""
    def __init__(self, message="Invalid input to the model.", *args):
        super().__init__(message, *args)

class InvalidParameterException(CustomException):
    """Exception raised for invalid hyperparameters or configuration."""
    def __init__(self, message="Invalid hyperparameters or configuration.", *args):
        super().__init__(message, *args)
