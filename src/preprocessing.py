import pandas as pd
import numpy as np
import yaml
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pickle

def get_params():
    try:
        with open("params.yaml", 'r') as file:
            params = yaml.safe_load(file)
            return params.get('preprocessing', {})
    except FileNotFoundError:
        return {}

def preprocess_data():
    # Get parameters
    params = get_params()
    train_path = params.get('train_path', 'data/california_housing_train.csv')
    test_path = params.get('test_path', 'data/california_housing_test.csv')
    
    # Read the data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate features and target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Create preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Fit the pipeline on training data and transform both train and test
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    
    # Save the pipeline
    os.makedirs('data', exist_ok=True)
    with open('data/preprocessing_pipeline.pkl', 'wb') as f:
        pickle.dump(preprocessing_pipeline, f)
    
    # Save the processed data
    pd.DataFrame(X_train_processed, columns=X_train.columns).to_csv('data/processed_train_features.csv', index=False)
    pd.DataFrame(X_test_processed, columns=X_test.columns).to_csv('data/processed_test_features.csv', index=False)
    y_train.to_csv('data/processed_train_target.csv', index=False)
    y_test.to_csv('data/processed_test_target.csv', index=False)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_data()