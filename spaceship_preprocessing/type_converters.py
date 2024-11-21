from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from .base import BaseSpaceshipTransformer

class DataTypeConverter(BaseSpaceshipTransformer):
    """Converts columns to appropriate dtypes.
    
    Handles:
    - Converting object columns to category
    - Converting string numbers to numeric
    - Special handling for cabin numbers
    - Memory optimization through appropriate dtype selection
    """
    
    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        category_threshold: float = 0.05,  # If unique values are less than 5% of total rows
        handle_cabin: bool = True
    ):
        """Initialize the converter.
        
        Args:
            categorical_columns: List of columns to force convert to categorical
            numeric_columns: List of columns to force convert to numeric
            category_threshold: Threshold for auto-converting to categorical (proportion of unique values)
            handle_cabin: Whether to handle cabin number conversion specially
        """
        super().__init__()
        self.categorical_columns = categorical_columns or []
        self.numeric_columns = numeric_columns or []
        self.category_threshold = category_threshold
        self.handle_cabin = handle_cabin
        
        # Will be set during fit
        self.detected_categorical_columns_: List[str] = []
        self.detected_numeric_columns_: List[str] = []
        self.unique_value_counts_: Dict[str, int] = {}
        
    def _detect_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """Detect columns that should be converted to categorical.
        
        A column is considered categorical if:
        1. It's explicitly specified in categorical_columns
        2. It's an object dtype and unique values are below threshold
        """
        categorical_cols = []
        n_rows = len(X)
        
        for col in X.columns:
            if col in self.categorical_columns:
                categorical_cols.append(col)
            elif X[col].dtype == 'object':
                n_unique = X[col].nunique()
                self.unique_value_counts_[col] = n_unique
                if n_unique / n_rows < self.category_threshold:
                    categorical_cols.append(col)
                    
        return categorical_cols
    
    def _detect_numeric_columns(self, X: pd.DataFrame) -> List[str]:
        """Detect columns that should be converted to numeric.
        
        Attempts to convert string columns to numeric if they contain
        valid numeric data.
        """
        numeric_cols = []
        
        for col in X.columns:
            if col in self.numeric_columns:
                numeric_cols.append(col)
            elif X[col].dtype == 'object':
                # Try converting to numeric
                try:
                    pd.to_numeric(X[col].dropna())
                    numeric_cols.append(col)
                except (ValueError, TypeError):
                    continue
                    
        return numeric_cols
    
    def _optimize_numeric_dtypes(self, series: pd.Series) -> pd.Series:
        """Optimize numeric dtype based on value range."""
        if series.dtype.kind in 'fc':  # float or complex
            return series.astype(np.float32)
        elif series.dtype.kind in 'iu':  # integer or unsigned
            if series.min() >= 0:
                if series.max() <= 255:
                    return series.astype(np.uint8)
                elif series.max() <= 65535:
                    return series.astype(np.uint16)
            else:
                if series.min() >= -128 and series.max() <= 127:
                    return series.astype(np.int8)
                elif series.min() >= -32768 and series.max() <= 32767:
                    return series.astype(np.int16)
        return series
    
    def _handle_cabin_number(self, X: pd.DataFrame) -> pd.DataFrame:
        """Special handling for cabin number conversion."""
        if 'Cabin' not in X.columns:
            return X
            
        X = X.copy()
        # Extract cabin number from the middle part of the cabin code
        X['Cabin_num'] = (X['Cabin']
                         .str.split('/', expand=True)[1]
                         .fillna('0')
                         .pipe(pd.to_numeric, errors='coerce')
                         .astype('float32'))
        return X
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataTypeConverter':
        """Fit the converter by detecting column types.
        
        Args:
            X: Input DataFrame
            y: Ignored
            
        Returns:
            self
        """
        self._validate_data(X)
        
        # Detect column types
        self.detected_categorical_columns_ = self._detect_categorical_columns(X)
        self.detected_numeric_columns_ = self._detect_numeric_columns(X)
        
        return super().fit(X, y)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by converting dtypes.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with converted dtypes
        """
        self._validate_is_fitted()
        X = X.copy()
        
        # Convert categorical columns
        for col in self.detected_categorical_columns_:
            if col in X.columns:
                X[col] = X[col].astype('category')
        
        # Convert numeric columns
        for col in self.detected_numeric_columns_:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = self._optimize_numeric_dtypes(X[col])
        
        # Handle cabin number if requested
        if self.handle_cabin:
            X = self._handle_cabin_number(X)
        
        return X
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names."""
        self._validate_is_fitted()
        base_columns = self.feature_names_in_
        if self.handle_cabin and 'Cabin' in base_columns:
            return list(base_columns) + ['Cabin_num']
        return list(base_columns)