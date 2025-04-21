from metaflow import FlowSpec, step, Parameter, Flow
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os

class ScoringFlow(FlowSpec):
    """Scoring flow for California Housing model"""
    
    model_name = Parameter('model_name', default='housing-model', help='Model name in registry', type=str)
    
    @step
    def start(self):
        """Start the flow"""
        print("Starting scoring flow")
        # Set local MLflow tracking
        os.makedirs("mlruns", exist_ok=True)
        mlflow.set_tracking_uri("file:./mlruns")
        self.next(self.load_model)
    
    @step
    def load_model(self):
        """Load the registered model"""
        print(f"Loading model: {self.model_name}")
        
        # Load model from local MLflow
        self.model = mlflow.sklearn.load_model(f"models:/{self.model_name}/latest")
        
        # Get scaler from training flow
        train_flow = Flow('TrainingFlow').latest_run
        self.scaler = train_flow['start'].task.data.scaler
        
        print("Model and scaler loaded successfully")
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """Load test data from training flow"""
        print("Loading test data from training flow")
        
        # Get data from training flow
        train_flow = Flow('TrainingFlow').latest_run
        self.X_test = train_flow['start'].task.data.X_test
        self.y_test = train_flow['start'].task.data.y_test
        
        print(f"Test data loaded: {self.X_test.shape[0]} samples")
        self.next(self.make_predictions)
    
    @step
    def make_predictions(self):
        """Make predictions with the model"""
        print("Making predictions")
        
        # Scale data
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Make predictions
        self.predictions = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        self.mse = mean_squared_error(self.y_test, self.predictions)
        self.r2 = r2_score(self.y_test, self.predictions)
        
        print(f"Predictions made. MSE: {self.mse:.4f}, R²: {self.r2:.4f}")
        self.next(self.end)
    
    @step
    def end(self):
        """End the flow"""
        print("Scoring flow completed successfully!")
        
        # Create a results DataFrame
        results = pd.DataFrame({
            'actual': self.y_test,
            'predicted': self.predictions,
            'error': self.y_test - self.predictions
        })
        
        # Display results summary
        print(f"MSE: {self.mse:.4f}")
        print(f"R²: {self.r2:.4f}")
        print(f"Sample predictions:\n{results.head()}")

if __name__ == '__main__':
    ScoringFlow()