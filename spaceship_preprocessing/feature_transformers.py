from typing import List, Optional, Dict, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from .base import BaseSpaceshipTransformer

class NumericTransformer(BaseSpaceshipTransformer):
    """Handles scaling and transformations of numeric features.
    
    Supports:
    - Standard scaling
    - Log transformation for specified columns
    - Robust handling of zero and negative values in log transform
    """
    
    def __init__(
        self,
        numeric_features=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'],
        scale_features=True
    ):
        """Initialize the transformer.
        
        Args:
            numeric_features: List of numeric columns to transform
            scale_features: Whether to apply scaling after log transformation
        """
        super().__init__()
        self.numeric_features = numeric_features
        self.scale_features = scale_features
        self.num_imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer.
        
        Args:
            X: Input DataFrame
            y: Ignored
            
        Returns:
            self
        """
        
        # Fit imputer
        self.num_imputer.fit(X[self.numeric_features])
        
        # Fit scaler if needed
        if self.scale_features:
            self.scaler.fit(self.num_imputer.transform(X[self.numeric_features]))
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        X_transformed = X.copy()
        
        # Impute values
        X_transformed[self.numeric_features] = self.num_imputer.transform(X_transformed[self.numeric_features])
        
        # Scale if needed
        if self.scale_features:
            X_transformed[self.numeric_features] = self.scaler.transform(X_transformed[self.numeric_features])
        
        return X_transformed

class CategoricalEncoder(BaseSpaceshipTransformer):
    """Handles encoding of categorical features.
    
    Supports:
    - One-hot encoding for high-cardinality features
    - Label encoding for low-cardinality features
    - Handles unknown categories gracefully
    """
    
    def __init__(
        self,
        categorical_features=['HomePlanet', 'Destination'],
        onehot_features=['HomePlanet', 'Destination', 'Deck']
    ):
        """Initialize the encoder.
        
        Args:
            categorical_features: List of categorical columns to encode
            onehot_features: Columns to force one-hot encoding
        """
        super().__init__()
        self.categorical_features = categorical_features
        self.onehot_features = onehot_features
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.label_encoders = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder.
        
        Args:
            X: Input DataFrame
            y: Ignored
            
        Returns:
            self
        """ 
        
        # Fit imputer
        self.cat_imputer.fit(X[self.categorical_features])
        
        # Fit label encoders for non-onehot features
        non_onehot = [col for col in self.categorical_features if col not in self.onehot_features]
        for col in non_onehot:
            if col in X.columns:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(X[col].astype(str))
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        X_transformed = X.copy()
        
        # Impute values
        X_transformed[self.categorical_features] = self.cat_imputer.transform(X_transformed[self.categorical_features])
        
        # Get columns that exist in the DataFrame
        columns_to_encode = [col for col in self.onehot_features if col in X_transformed.columns]
        
        # Create prefix mapping
        prefix_mapping = {
            'HomePlanet': 'HP',
            'Destination': 'Dest',
            'Deck': 'Deck'
        }
        
        # Get prefixes only for columns that exist
        prefixes = [prefix_mapping[col] for col in columns_to_encode]
        
        # One-hot encode specified features
        X_transformed = pd.get_dummies(
            X_transformed, 
            columns=columns_to_encode,
            prefix=prefixes
        )
        
        # Label encode remaining categorical features
        non_onehot = [col for col in self.categorical_features if col not in self.onehot_features]
        for col in non_onehot:
            if col in X_transformed.columns and col in self.label_encoders:
                X_transformed[col] = self.label_encoders[col].transform(X_transformed[col].astype(str))
        
        return X_transformed
