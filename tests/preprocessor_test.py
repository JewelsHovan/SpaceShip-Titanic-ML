from spaceship_preprocessing.pipeline import SpaceshipPreprocessor

# Initialize the preprocessor with desired settings
preprocessor = SpaceshipPreprocessor(
    handle_unknown_categories='ignore',
    n_cabin_groups=5,
    add_location_features=True
)

# Get preprocessed training and test data
X, X_test, y = preprocessor.preprocess_data(
    train_data_path='data/train.csv',
    test_data_path='data/test.csv'
)

# Print information about the transformed data
print("\nProcessed Data Info:")
print("-------------------")
print("Training set shape:", X.shape)
print("Test set shape:", X_test.shape)
print("Target variable shape:", y.shape)

# Check for missing values
print("\nMissing Values:")
print("Training set:", X.isnull().sum().sum())
print("Test set:", X_test.isnull().sum().sum())
print("Target variable:", y.isnull().sum())

# Get and print feature names
feature_names = preprocessor.get_feature_names()
print("\nFeatures:", feature_names)