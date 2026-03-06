
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import pickle
from sklearn.pipeline import Pipeline
import pandas as pd


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer.
    
    This class inherits from:
    - BaseEstimator: This helps make the class compatible with scikit-learn's estimator API.
    - TransformerMixin: This provides the fit_transform method, which combines fit and transform.

    Purpose:
    - This helps in creating new features or modifying existing ones in the dataset.
    - It stops the risk of data leakage by ensuring that feature engineering is done within the pipeline.

    BaseEstimator: makes the class parameters easily accessible and allows for hyperparameter tuning.
    TransformerMixin: provides the fit_transform method for convenience.
    """
    
    def __init__(self) -> None:
        """
        Constructor to initialize any parameters if needed.
        """
        pass

    def fit(self, X: pd.DataFrame, y=None) -> "FeaturePipeline":
        """
        Fit step to learn any parameters from the training data.
        """
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the feature transformations to the data.
        """
        pass

    def _feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies only the feature engineering steps without preprocessing.
        Useful for MI score calculations.
        """
        pass


