from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from .base import BaseSpaceshipTransformer
from sklearn.preprocessing import LabelEncoder

class CabinFeatureEngineer(BaseSpaceshipTransformer):
    """Engineers features from Cabin information.
    
    Creates:
    - Deck features (one-hot encoded)
    - Cabin number groups (quantile-based)
    - Side features (P/S binary encoding)
    """
    
    def __init__(
        self,
        n_cabin_groups: int = 5,
        add_location_features: bool = True
    ):
        """Initialize the engineer.
        
        Args:
            n_cabin_groups: Number of quantile groups for cabin numbers
            add_location_features: Whether to add front/middle/back location features
        """
        super().__init__()
        self.n_cabin_groups = n_cabin_groups
        self.add_location_features = add_location_features
        
        # Will be set during fit
        self.cabin_group_bins_: List[float] = []
        self.deck_values_: List[str] = []
        self.label_encoder_ = LabelEncoder()
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CabinFeatureEngineer':
        """Fit the engineer.
        
        Args:
            X: Input DataFrame
            y: Ignored
            
        Returns:
            self
        """
        self._validate_data(X)
        
        # Extract cabin components
        cabin_df = self._split_cabin(X)
        
        # Get unique deck values
        self.deck_values_ = sorted(cabin_df['Deck'].unique())
        
        # Calculate cabin number quantile bins
        cabin_nums = cabin_df['Cabin_num'].fillna(cabin_df['Cabin_num'].median())
        self.cabin_group_bins_ = pd.qcut(
            cabin_nums,
            q=self.n_cabin_groups,
            retbins=True,
            duplicates='drop'
        )[1]
        
        # Fit label encoder on group labels including Unknown
        group_labels = [f'Group_{i+1}' for i in range(len(self.cabin_group_bins_)-1)] + ['Unknown']
        self.label_encoder_.fit(group_labels)
        
        self._fitted = True
        
        # Transform once during fit to establish feature names
        X_transformed = self.transform(X)
        self.feature_names_out_ = X_transformed.columns.tolist()
        
        return self
    
    def _split_cabin(self, X: pd.DataFrame) -> pd.DataFrame:
        """Split cabin into components."""
        df = pd.DataFrame()
        
        # Handle missing values
        cabin_split = X['Cabin'].fillna('Unknown/0/Unknown').str.split('/', expand=True)
        
        df['Deck'] = cabin_split[0]
        df['Cabin_num'] = pd.to_numeric(cabin_split[1], errors='coerce')
        df['Side'] = cabin_split[2]
        
        return df
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        self._validate_is_fitted()
        X_transformed = X.copy()
        
        # Handle missing Cabin values
        X_transformed['Cabin'].fillna('Unknown/0/Unknown', inplace=True)
        
        # Split cabin into components
        cabin_split = X_transformed['Cabin'].str.split('/', expand=True)
        X_transformed['Deck'] = cabin_split[0]
        X_transformed['Side'] = cabin_split[2]
        
        # Handle cabin number with binning
        X_transformed['Cabin_num'] = pd.to_numeric(cabin_split[1].fillna('0'), errors='coerce')
        
        # Create cabin number groups and encode them numerically
        cabin_nums = X_transformed['Cabin_num'].fillna(X_transformed['Cabin_num'].median())
        
        # First create the groups with an additional bin for Unknown
        group_labels = [f'Group_{i+1}' for i in range(len(self.cabin_group_bins_)-1)]
        
        # Create categorical groups with ordered=False to allow adding new categories
        groups = pd.cut(
            cabin_nums,
            bins=self.cabin_group_bins_,
            labels=group_labels,
            ordered=False  # Allow adding new categories
        )
        
        # Convert to string type before handling unknown values
        groups = groups.astype(str)
        groups = groups.replace('nan', 'Unknown')

        # Then encode them numerically
        X_transformed['Cabin_Number_Group'] = self.label_encoder_.transform(groups)
        
        # Create deck features
        deck_dummies = pd.get_dummies(X_transformed['Deck'], prefix='Deck')
        X_transformed = pd.concat([X_transformed, deck_dummies], axis=1)
        
        # Binary encode Side
        X_transformed['Cabin_Side_P'] = (X_transformed['Side'] == 'P').astype(int)
        
        # Add location features if enabled
        if self.add_location_features:
            X_transformed['Cabin_Front'] = (cabin_nums <= cabin_nums.quantile(0.33)).astype(int)
            X_transformed['Cabin_Back'] = (cabin_nums >= cabin_nums.quantile(0.67)).astype(int)
        
        # Drop intermediate columns
        columns_to_drop = ['Cabin', 'Deck', 'Side', 'Cabin_num']
        X_transformed.drop(columns=[col for col in columns_to_drop if col in X_transformed.columns], 
                         inplace=True)
        
        return X_transformed

class SpendingFeatureEngineer(BaseSpaceshipTransformer):
    """Engineers features from spending columns.
    
    Creates:
    - Total spending
    - Spending patterns (ratios)
    - Log-transformed spending
    """
    
    def __init__(
        self,
        spending_cols: Optional[List[str]] = None,
        add_ratios: bool = True
    ):
        """Initialize the engineer.
        
        Args:
            spending_cols: List of spending columns
            add_ratios: Whether to add spending ratio features
        """
        super().__init__()
        self.spending_cols = spending_cols or [
            'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'
        ]
        self.add_ratios = add_ratios
        # Will be set during fit
        self.spending_medians_: Dict[str, float] = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'SpendingFeatureEngineer':
        """Fit the engineer by calculating medians for imputation."""
        self._validate_data(X)
        
        # Calculate and store medians for each spending column
        for col in self.spending_cols:
            self.spending_medians_[col] = X[col].median()
        
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        self._validate_is_fitted()
        X_transformed = X.copy()
        
        # Fill missing values with medians
        for col in self.spending_cols:
            X_transformed[col] = X_transformed[col].fillna(self.spending_medians_[col])
        
        # Total spending
        X_transformed['TotalSpending'] = X_transformed[self.spending_cols].sum(axis=1)
        
        # Impute TotalSpending missing values with median
        total_spending_median = X_transformed['TotalSpending'].median()
        X_transformed['TotalSpending'] = X_transformed['TotalSpending'].fillna(total_spending_median)
        
        # Spending ratios
        if self.add_ratios and len(self.spending_cols) > 1:
            total_spending = X_transformed['TotalSpending'].replace(0, 1)  # Avoid division by zero
            for col in self.spending_cols:
                X_transformed[f'{col}_Ratio'] = X_transformed[col] / total_spending
        
        # Log transform spending
        for col in self.spending_cols + ['TotalSpending']:
            X_transformed[f'{col}_Log'] = np.log1p(X_transformed[col])
            # Ensure log transformed values don't have missing values
            log_col_median = X_transformed[f'{col}_Log'].median()
            X_transformed[f'{col}_Log'] = X_transformed[f'{col}_Log'].fillna(log_col_median)
    
        return X_transformed

class PassengerFeatureEngineer(BaseSpaceshipTransformer):
    """Engineers features from passenger information."""
    
    def __init__(
        self,
        age_bins: Optional[List[float]] = None,
        add_vip_features: bool = True
    ):
        super().__init__()
        self.age_bins = age_bins or [0, 18, 30, 45, 60, np.inf]
        self.add_vip_features = add_vip_features
        # Will be set during fit
        self.age_group_encoder_ = LabelEncoder()
        self.age_median_: float = 0
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PassengerFeatureEngineer':
        """Fit the engineer."""
        self._validate_data(X)
        
        # Calculate and store age median
        self.age_median_ = X['Age'].median()
        
        # Fit age group encoder
        age_groups = pd.cut(
            X['Age'].fillna(self.age_median_),
            bins=self.age_bins,
            labels=[f'AgeGroup_{i}' for i in range(len(self.age_bins)-1)],
            ordered=False
        )
        age_groups = age_groups.astype(str)
        age_groups = age_groups.replace('nan', 'Unknown')
        self.age_group_encoder_.fit(list(age_groups.unique()) + ['Unknown'])
        
        self._fitted = True
        
        # Transform once during fit to establish feature names
        X_transformed = self.transform(X)
        self.feature_names_out_ = X_transformed.columns.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        self._validate_is_fitted()
        X_transformed = X.copy()
        
        # Extract group from PassengerId and calculate group size
        X_transformed['Group'] = X_transformed['PassengerId'].str.split('_').str[0]
        X_transformed['GroupSize'] = X_transformed.groupby('Group')['Group'].transform('count')
        X_transformed['IsSolo'] = (X_transformed['GroupSize'] == 1).astype(int)
        
        # Age groups with proper handling of missing values and categorical encoding
        age_groups = pd.cut(
            X_transformed['Age'].fillna(self.age_median_),
            bins=self.age_bins,
            labels=[f'AgeGroup_{i}' for i in range(len(self.age_bins)-1)],
            ordered=False
        )
        # Convert to string and handle unknown values
        age_groups = age_groups.astype(str)
        age_groups = age_groups.replace('nan', 'Unknown')
        
        # Encode age groups numerically
        X_transformed['AgeGroup'] = self.age_group_encoder_.transform(age_groups)
        
        # VIP status features
        if self.add_vip_features:
            if 'VIP' in X_transformed.columns:
                X_transformed['VIP'] = X_transformed['VIP'].fillna(0).astype(int)
            else:
                X_transformed['VIP'] = 0
        
        # Clean up intermediate columns
        columns_to_drop = ['Group', 'PassengerId']
        X_transformed.drop(columns=[col for col in columns_to_drop if col in X_transformed.columns], 
                         inplace=True)
        
        return X_transformed

class NameFeatureEngineer(BaseSpaceshipTransformer):
    """Engineers features from passenger names.
    
    Creates:
    - First name initial
    - Last name initial
    """
    
    def __init__(self):
        """Initialize the engineer."""
        super().__init__()
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with engineered name features
        """
        self._validate_is_fitted()
        X_transformed = X.copy()
        
        # Handle missing names
        X_transformed['Name'].fillna('Unknown Unknown', inplace=True)
        
        # Split name into components
        name_parts = X_transformed['Name'].str.split(' ', expand=True)
        
        # Extract initials
        X_transformed['FirstNameInitial'] = name_parts[0].str[0]
        X_transformed['LastNameInitial'] = name_parts[1].str[0]
        
        # Drop the original Name column
        X_transformed.drop('Name', axis=1, inplace=True)
        
        # Store output feature names
        self.feature_names_out_ = X_transformed.columns.tolist()
        
        return X_transformed