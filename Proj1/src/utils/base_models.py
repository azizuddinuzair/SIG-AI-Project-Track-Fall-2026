
class BaseModel:
    """
    This is a parent class for different ML models. 
    It holds a pipeline that includes feature engineering, preprocessing, and the model itself.
    
    Purpose:
    - This structure helps to avoid data leakage by ensuring that all transformations are applied consistently.
    - It also makes the code more organized and easier to maintain.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        """
        Constructor to initialize the model with a given pipeline.
        """
        self.pipeline = pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """
        Train the model using the provided data (it should be training data).
        """
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        Returns an array of predicted labels.
        """
        return self.pipeline.predict(X)
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk using pickle.
        
        Args:
            filepath: Path where the model should be saved (should end with .pkl)
        """
        pass
    
    @classmethod
    def load(cls, filepath: str) -> "BaseModel":
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file (.pkl)
            
        Returns:
            An instance of the model class with the loaded pipeline
        """
        pass