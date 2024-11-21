import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        # Initialize imputers
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        
        # Initialize transformers
        self.scaler = StandardScaler()
        self.group_size_scaler = StandardScaler()
        self.label_encoders = {}
        
        # Add a final imputer for any remaining NaN values
        self.final_imputer = SimpleImputer(strategy='constant', fill_value=0)
        
    def _handle_missing_values(self, df, is_training=True):
        """Handle missing values and create missing indicators"""
        # Numeric features
        numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        
        # Create missing indicators
        for col in numeric_features:
            df[f'{col}_Missing'] = df[col].isna().astype(int)
        
        # Impute numeric features
        if is_training:
            df[numeric_features] = self.num_imputer.fit_transform(df[numeric_features])
        else:
            df[numeric_features] = self.num_imputer.transform(df[numeric_features])
        
        # Scale numeric features
        if is_training:
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        else:
            df[numeric_features] = self.scaler.transform(df[numeric_features])
        
        # Categorical features
        categorical_features = ['HomePlanet', 'Destination']
        
        # Create missing indicators
        for col in categorical_features:
            df[f'{col}_Missing'] = df[col].isna().astype(int)
        
        # Impute categorical features
        if is_training:
            df[categorical_features] = self.cat_imputer.fit_transform(df[categorical_features])
        else:
            df[categorical_features] = self.cat_imputer.transform(df[categorical_features])
        
        return df
    
    def _preprocess_common(self, df, is_training=True):
        """Common preprocessing steps for both training and test data"""
        # 1. Handle missing values first
        df = self._handle_missing_values(df, is_training)
        
        # 2. Create derived features
        df = self._split_cabin(df)
        df = self._extract_passenger_info(df, is_training)
        df = self._handle_spending_features(df)
        df = self._handle_age_features(df, is_training)
        
        # 3. Handle categorical encoding
        df = self._handle_categorical_features(df, is_training)
        
        # 4. Drop unnecessary columns
        columns_to_drop = ['PassengerId', 'Name', 'Cabin', 'Side', 'Cabin_num', 'Group']
        df.drop([col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)
        
        # Add final imputation step before returning
        if is_training:
            df = pd.DataFrame(
                self.final_imputer.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
        else:
            df = pd.DataFrame(
                self.final_imputer.transform(df),
                columns=df.columns,
                index=df.index
            )
        
        return df
    
    def _handle_spending_features(self, df):
        # Create spending features as recommended
        spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        
        # Total spending
        df['TotalSpending'] = df[spending_cols].sum(axis=1)
        
        # Binary spending flags
        df['HasAnySpending'] = (df['TotalSpending'] > 0).astype(int)
        
        # Log transform for non-zero values
        for col in spending_cols + ['TotalSpending']:
            df[f'{col}_Log'] = np.log1p(df[col])
        
        return df
    
    def _handle_age_features(self, df, is_training=True):
        # Add missing value flag for Age
        df['Age_Missing'] = df['Age'].isna().astype(int)
        
        # Create age groups
        df['AgeGroup'] = pd.cut(
            df['Age'].fillna(df['Age'].median()),  # Temporarily fill NaN for grouping
            bins=[0, 12, 18, 25, 35, 50, 100],
            labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']
        )
        
        # One-hot encode age groups
        df = pd.get_dummies(df, columns=['AgeGroup'], prefix='Age')
        
        return df
    
    def _split_cabin(self, df):
        # Handle missing Cabin values
        df['Cabin'].fillna('Unknown/0/Unknown', inplace=True)
        
        # Split Cabin into components
        cabin_split = df['Cabin'].str.split('/', expand=True)
        df['Deck'] = cabin_split[0]
        df['Side'] = cabin_split[2]
        
        # Handle cabin number with binning
        df['Cabin_num'] = cabin_split[1].fillna('0')
        df['Cabin_num'] = pd.to_numeric(df['Cabin_num'], errors='coerce')
        df['Cabin_Number_Group'] = pd.qcut(
            df['Cabin_num'].fillna(df['Cabin_num'].median()), 
            q=5, 
            labels=['G1', 'G2', 'G3', 'G4', 'G5']
        )
        
        # Binary encode Side
        df['Cabin_Side_P'] = (df['Side'] == 'P').astype(int)
        
        return df
    
    def _extract_passenger_info(self, df, is_training=True):
        df['Group'] = df['PassengerId'].str.split('_').str[0]
        df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
        df['IsSolo'] = (df['GroupSize'] == 1).astype(int)
        
        # Scale GroupSize
        if is_training:
            df['GroupSize'] = self.group_size_scaler.fit_transform(df[['GroupSize']])
        else:
            df['GroupSize'] = self.group_size_scaler.transform(df[['GroupSize']])
        
        return df
    
    def _handle_categorical_features(self, df, is_training=True):
        """Handle categorical feature encoding"""
        # One-hot encode high-impact categoricals
        categorical_features = ['HomePlanet', 'Destination']
        if 'Deck' in df.columns:
            categorical_features.append('Deck')
        
        df = pd.get_dummies(df, columns=categorical_features, prefix=['HP', 'Dest', 'Deck'])
        
        # Label encode remaining categorical features
        if 'Cabin_Number_Group' in df.columns:
            if is_training:
                self.label_encoders['Cabin_Number_Group'] = LabelEncoder()
                df['Cabin_Number_Group'] = self.label_encoders['Cabin_Number_Group'].fit_transform(df['Cabin_Number_Group'])
            else:
                df['Cabin_Number_Group'] = self.label_encoders['Cabin_Number_Group'].transform(df['Cabin_Number_Group'])
        
        return df
    
    def preprocess_data(self, data_path, test_size=0.2, random_state=42):
        df = pd.read_csv(data_path)
        
        if 'Transported' in df.columns:
            y = df['Transported'].astype(int)
            df.drop('Transported', axis=1, inplace=True)
        else:
            y = None
        
        # Apply preprocessing
        df = self._preprocess_common(df, is_training=True)
        
        if y is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                df, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
            return X_train, X_val, y_train, y_val
        else:
            return df, None, None, None
    
    def preprocess_test_data(self, test_data_path):
        df = pd.read_csv(test_data_path)
        passenger_ids = df['PassengerId'].copy()
        
        # Apply preprocessing
        df = self._preprocess_common(df, is_training=False)
        
        return df, passenger_ids

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Preprocess training data
    X_train, X_val, y_train, y_val = preprocessor.preprocess_data('data/train.csv')
    
    # Print shapes to verify
    print("\nTraining Data Info:")
    print("------------------")
    print("Training set shape:", X_train.shape)
    print("Validation set shape:", X_val.shape)
    print("Training labels shape:", y_train.shape)
    print("Validation labels shape:", y_val.shape)
    
    # Print feature names
    print("\nFeatures:", list(X_train.columns))
