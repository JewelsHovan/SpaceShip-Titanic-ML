# Spaceship Titanic Feature Analysis Report

## Key Features Overview

### High-Impact Features

1. **CryoSleep** (Binary)
   - Strong predictor of transportation (p-value < 0.0001)
   - Clear difference in transportation rates:
     - True: ~74% transported
     - False: ~33% transported
   - Recommendation: Use as-is, binary encoding

2. **Cabin** (Categorical - Engineered)
   - Successfully split into Deck/Number/Side
   - **Deck** shows significant patterns (p-value < 0.0001)
   - **Side** shows minor differences (P vs S)
   - **Number** shows statistical significance
   - Recommendation: 
     - Use Deck as categorical feature
     - Consider binning Number into groups
     - Side could be binary encoded

3. **Age** (Numerical)
   - Significant correlation with transportation (p-value < 0.0001)
   - Created age groups show clear patterns
   - Recommendation:
     - Use both raw age and age groups
     - Age groups: Child/Teen/Young Adult/Adult/Middle Aged/Senior

4. **Spending Features** (Numerical)
   - All spending features show significant patterns
   - High zero-spending rates across features
   - Recommendation:
     - Create aggregate spending feature
     - Create binary flags for "any spending"
     - Use spending level groups (Very Low to Very High)
     - Consider log transformation for non-zero values

### Moderate-Impact Features

1. **HomePlanet** (Categorical)
   - Three categories with different transportation rates
   - Statistically significant (p-value < 0.0001)
   - Recommendation: One-hot encoding

2. **Destination** (Categorical)
   - Three destinations with varying transportation rates
   - Statistically significant differences
   - Recommendation: One-hot encoding

3. **VIP** (Binary)
   - Shows some correlation with transportation
   - Recommendation: Use as-is, binary encoding

### Low-Impact Features

1. **Name** (Categorical)
   - First/Last initials show some patterns
   - High cardinality
   - Recommendation:
     - Drop original name
     - Consider using initials if model capacity allows
     - Could be excluded without significant impact

## Feature Engineering Recommendations

1. **Spending Features Transformation**:
```python
# Aggregate spending
df['TotalSpending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)

# Binary spending flags
df['HasAnySpending'] = df['TotalSpending'] > 0

# Log transform for non-zero values
spending_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpending']
for col in spending_columns:
    df[f'{col}_Log'] = np.log1p(df[col])
```

2. **Cabin Feature Engineering**:
```python
# Deck encoding
df_encoded = pd.get_dummies(df['Deck'], prefix='Deck')

# Number binning
df['Cabin_Number_Group'] = pd.qcut(df['Number'], q=5, labels=['G1', 'G2', 'G3', 'G4', 'G5'])

# Side binary encoding
df['Cabin_Side_P'] = (df['Side'] == 'P').astype(int)
```

3. **Age Processing**:
```python
# Keep raw age and add age groups
df['AgeGroup'] = pd.cut(df['Age'], 
                       bins=[0, 12, 18, 25, 35, 50, 100],
                       labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Middle Aged', 'Senior'])
```

## Missing Data Strategy

- Missing data appears to be MCAR (Missing Completely at Random)
- Recommendation:
  1. For numerical features: Impute with median
  2. For categorical features: Impute with mode
  3. Add binary flags for missingness
  4. Consider using more sophisticated imputation methods like KNN or iterative imputation

## Final Feature Set Recommendation

1. **Core Features**:
   - CryoSleep (binary)
   - Deck (one-hot encoded)
   - Cabin_Side_P (binary)
   - Cabin_Number_Group (categorical)
   - Age (numerical)
   - AgeGroup (categorical)
   - HomePlanet (one-hot encoded)
   - Destination (one-hot encoded)
   - VIP (binary)

2. **Engineered Features**:
   - TotalSpending_Log
   - HasAnySpending
   - Individual spending features (log transformed)
   - Missing data flags

3. **Optional Features**:
   - Name initials
   - Raw cabin numbers
   - Individual spending categories (non-transformed)

This feature set should provide a strong foundation for modeling while maintaining interpretability and capturing the most important patterns in the data.