from typing import Tuple, Optional, List
import pandas as pd
from sklearn.pipeline import Pipeline

from .feature_transformers import NumericTransformer, CategoricalEncoder
from .feature_engineers import PassengerFeatureEngineer, CabinFeatureEngineer, NameFeatureEngineer, SpendingFeatureEngineer
from .type_converters import DataTypeConverter
from .imputers import NumericImputer, CategoricalImputer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class SpaceshipPreprocessor:
    """Main preprocessor that combines all transformers in a pipeline.
    
    Handles the complete preprocessing workflow:
    1. Numeric feature transformation
    2. Cabin feature engineering
    3. Categorical feature encoding
    4. Feature engineering
    """
    
    def __init__(
        self,
        handle_unknown_categories: str = 'ignore',
        n_cabin_groups: int = 5,
        add_location_features: bool = True
    ):
        """Initialize the preprocessor.
        
        Args:
            handle_unknown_categories: How to handle unknown categories ('ignore' or 'error')
            n_cabin_groups: Number of quantile groups for cabin numbers
            add_location_features: Whether to add location features based on cabin numbers
        """
        self.pipeline = Pipeline([
            # Step 1: Convert data types
            ('type_converter', DataTypeConverter(
                handle_cabin=True,
                category_threshold=0.05
            )),
            
            # Step 2: Handle name features
            ('name_engineer', NameFeatureEngineer()),
            
            # Step 3: Handle numeric imputation
            ('numeric_imputer', NumericImputer(
                strategy='median',
                columns=['Age', 'RoomService', 'FoodCourt', 
                        'ShoppingMall', 'Spa', 'VRDeck'],
                add_indicators=True
            )),
            
        # Step 4: Handle categorical imputation
            ('categorical_imputer', CategoricalImputer(
                strategy='most_frequent',
                columns=['HomePlanet', 'Destination', 'CryoSleep', 'VIP'],
                add_indicators=True
            )),
            
            # Step 5: Handle numeric features
            ('numeric', NumericTransformer(
                numeric_features=['Age', 'RoomService', 'FoodCourt', 
                                'ShoppingMall', 'Spa', 'VRDeck'],
                scale_features=True
            )),
            
            # Step 6: Handle Cabin features
            ('cabin', CabinFeatureEngineer(
                n_cabin_groups=n_cabin_groups,
                add_location_features=add_location_features
            )),
            
            # Step 7: Handle categorical features
            ('categorical', CategoricalEncoder(
                categorical_features=['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'FirstNameInitial', 'LastNameInitial'],
                onehot_features=['HomePlanet', 'Destination']
            )),
            
            # Step 8: Add spending features
            ('spending', SpendingFeatureEngineer()),
            
            # Step 9: Engineer remaining features
            ('feature_engineering', PassengerFeatureEngineer())
        ])
        
        self.handle_unknown = handle_unknown_categories
        self.n_cabin_groups = n_cabin_groups
        self.add_location_features = add_location_features
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit the pipeline and transform the data.
        
        Args:
            X: Input DataFrame
            y: Target variable (optional)
            
        Returns:
            Transformed DataFrame
        """
        return self.pipeline.fit_transform(X, y)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted pipeline.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        return self.pipeline.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get the names of the output features.
        
        Returns:
            List of feature names
        """
        # Get the final transformed DataFrame's columns
        return self.pipeline.named_steps['feature_engineering'].get_feature_names_out()
    
    def preprocess_data(
        self,
        train_data_path: str,
        test_data_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.Series]]:
        """Preprocess both training and test data.
        
        Args:
            train_data_path: Path to training data CSV
            test_data_path: Path to test data CSV (optional)
            
        Returns:
            Tuple of (transformed train data, transformed test data, target variable)
        """
        # Load training data
        train_df = pd.read_csv(train_data_path)
        
        # Store PassengerId if present for test data
        passenger_ids = None
        if test_data_path:
            test_df = pd.read_csv(test_data_path)
            passenger_ids = test_df['PassengerId'].copy()
        
        # Separate target variable if present
        if 'Transported' in train_df.columns:
            y = train_df['Transported'].astype(int)
            X = train_df.drop('Transported', axis=1)
        else:
            y = None
            X = train_df
        
        # Fit and transform training data
        X_transformed = self.fit_transform(X, y)
        
        # Transform test data if provided
        if test_data_path:
            X_test_transformed = self.transform(test_df)
            # Add PassengerId back if needed
            if passenger_ids is not None:
                X_test_transformed['PassengerId'] = passenger_ids
        else:
            X_test_transformed = None
        
        return X_transformed, X_test_transformed, y
