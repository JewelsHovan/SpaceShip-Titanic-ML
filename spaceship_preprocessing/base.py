from typing import List, Union, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class BaseSpaceshipTransformer(BaseEstimator, TransformerMixin):
    """Base class for all spaceship transformers with common utilities.
    
    Provides common functionality for data validation, column handling,
    and basic transformer operations.
    """
    
    def __init__(self, columns: Optional[List[str]] = None):
        """Initialize the transformer.
        
        Args:
            columns: List of column names to transform. If None, will process all suitable columns.
        """
        self.columns = columns
        self.feature_names_in_ = None
        self._fitted = False
    
    def _validate_data(self, X: pd.DataFrame) -> None:
        """Validate input data.
        
        Args:
            X: Input DataFrame to validate
            
        Raises:
            TypeError: If X is not a pandas DataFrame
            ValueError: If required columns are missing
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
            
        if self.columns is not None:
            missing_cols = set(self.columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _validate_is_fitted(self) -> None:
        """Check if the transformer is fitted.
        
        Raises:
            ValueError: If the transformer is not fitted
        """
        if not self._fitted:
            raise ValueError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names.
        
        Returns:
            List of output feature names
            
        Raises:
            ValueError: If transformer is not fitted
        """
        self._validate_is_fitted()
        return self.feature_names_out_
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseSpaceshipTransformer':
        """Fit the transformer.
        
        Args:
            X: Input DataFrame
            y: Target variable (optional)
            
        Returns:
            self
        """
        self._validate_data(X)
        self.feature_names_in_ = X.columns.tolist()
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        self._validate_is_fitted()
        self._validate_data(X)
        return X.copy()
    
    def get_columns(self, X: pd.DataFrame, dtype: Optional[Union[str, List[str]]] = None) -> List[str]:
        """Get columns of specified dtype.
        
        Args:
            X: Input DataFrame
            dtype: Specific dtype(s) to filter columns by
            
        Returns:
            List of column names
        """
        if self.columns is not None:
            return self.columns
            
        if dtype is None:
            return X.columns.tolist()
            
        dtypes = [dtype] if isinstance(dtype, str) else dtype
        return [col for col, col_dtype in X.dtypes.items() 
                if any(d in str(col_dtype) for d in dtypes)]