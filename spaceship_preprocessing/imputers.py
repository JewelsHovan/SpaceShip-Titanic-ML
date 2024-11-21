from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from .base import BaseSpaceshipTransformer

class NumericImputer(BaseSpaceshipTransformer):
    """Handles numeric feature imputation with missing indicators.
    
    Imputes numeric features using specified strategy
    """
    
    def __init__(
        self,
        strategy: str = 'median',
        columns: Optional[List[str]] = None,
        add_indicators: bool = False
    ):
        """Initialize the numeric imputer.
        
        Args:
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
            columns: List of numeric columns to impute. If None, will detect numeric columns
            add_indicators: Whether to add binary missing value indicators
        """
        super().__init__(columns)
        self.strategy = strategy
        self.add_indicators = add_indicators
        self.imputer = SimpleImputer(strategy=strategy)
        self.feature_names_out_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'NumericImputer':
        """Fit the imputer on the input data.
        
        Args:
            X: Input DataFrame
            y: Target variable (unused)
            
        Returns:
            self
        """
        self._validate_data(X)
        
        # Get numeric columns if not specified
        self.columns = self.columns or self.get_columns(X, dtype=['int64', 'float64'])
        
        # Fit the imputer
        self.imputer.fit(X[self.columns])
        
        # Simplified feature names - no more indicators
        self.feature_names_out_ = self.columns
        
        return super().fit(X, y)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with imputed values and optional missing indicators
        """
        self._validate_is_fitted()
        X_copy = X.copy()
        
        # Perform imputation
        X_copy[self.columns] = self.imputer.transform(X_copy[self.columns])
        
        return X_copy

class CategoricalImputer(BaseSpaceshipTransformer):
    """Handles categorical feature imputation with missing indicators.
    
    Imputes categorical features using specified strategy and creates binary flags
    for missing values.
    """
    
    def __init__(
        self,
        strategy: str = 'most_frequent',
        columns: Optional[List[str]] = None,
        add_indicators: bool = False,
        fill_value: Optional[str] = None,
        handle_unknown: str = 'error'
    ):
        """Initialize the categorical imputer.
        
        Args:
            strategy: Imputation strategy ('most_frequent', 'constant')
            columns: List of categorical columns to impute. If None, will detect object/category columns
            add_indicators: Whether to add binary missing value indicators
            fill_value: Value to use when strategy is 'constant'
            handle_unknown: Ignored parameter for pipeline compatibility
        """
        super().__init__(columns)
        self.strategy = strategy
        self.add_indicators = add_indicators
        self.fill_value = fill_value
        self.handle_unknown = handle_unknown
        self.imputer = SimpleImputer(
            strategy=strategy,
            fill_value=fill_value
        )
        self.feature_names_out_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CategoricalImputer':
        """Fit the imputer on the input data.
        
        Args:
            X: Input DataFrame
            y: Target variable (unused)
            
        Returns:
            self
        """
        self._validate_data(X)
        
        # Get categorical columns if not specified
        self.columns = self.columns or self.get_columns(X, dtype=['object', 'category'])
        
        # Fit the imputer
        self.imputer.fit(X[self.columns])
        
        # Simplified feature names - no more indicators
        self.feature_names_out_ = self.columns
        
        return super().fit(X, y)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with imputed values and optional missing indicators
        """
        self._validate_is_fitted()
        X_copy = X.copy()
        
        # Perform imputation
        X_copy[self.columns] = self.imputer.transform(X_copy[self.columns])
        
        return X_copy