from metaflow import FlowSpec, step, Parameter, Flow
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os

class ScoringFlow(FlowSpec):
    """
    Uses the model trained in TrainingFlow to make predictions.
    
    Steps:
    1. Load the model from MLflow
    2. Get test data from the training flow
    3. Make predictions
    4. Check how well we did
    """
    
    # Parameter we can change at runtime
    model_name = Parameter('model_name', default='housing-model', 
                          help='Name of the model in MLflow', type=str)
    
    @step
    def start(self):
        """
        Sets up MLflow and starts the flow.
        """
        print("Starting scoring flow")
        
        # Make sure MLflow directory exists
        os.makedirs("mlruns", exist_ok=True)
        
        # Point to local MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        
        self.next(self.load_model)
    
    @step
    def load_model(self):
        """
        Loads the model from MLflow and gets the scaler
        from the training flow.
        """
        print(f"Loading model: {self.model_name}")
        
        # Get latest version of the model
        self.model = mlflow.sklearn.load_model(f"models:/{self.model_name}/latest")
        
        # Get the scaler from training - need this to transform new data
        train_flow = Flow('TrainingFlow').latest_run
        self.scaler = train_flow['start'].task.data.scaler
        
        print("Model and scaler loaded successfully")
        
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """
        Gets test data from the latest training run.
        Shows how we can access data between flows.
        """
        print("Loading test data from training flow")
        
        # Get the test data that was held out during training
        train_flow = Flow('TrainingFlow').latest_run
        self.X_test = train_flow['start'].task.data.X_test
        self.y_test = train_flow['start'].task.data.y_test
        
        print(f"Test data loaded: {self.X_test.shape[0]} samples")
        
        self.next(self.make_predictions)
    
    @step
    def make_predictions(self):
        """
        Uses the model to make predictions and
        calculates how well we did.
        """
        print("Making predictions")
        
        # Scale data the same way we did during training
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Get predictions
        self.predictions = self.model.predict(X_test_scaled)
        
        # How'd we do?
        self.mse = mean_squared_error(self.y_test, self.predictions)
        self.r2 = r2_score(self.y_test, self.predictions)
        
        print(f"Predictions made. MSE: {self.mse:.4f}, R²: {self.r2:.4f}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Wraps up and shows some results.
        """
        print("Scoring flow completed successfully!")
        
        # Put results in a DataFrame for easy viewing
        results = pd.DataFrame({
            'actual': self.y_test,
            'predicted': self.predictions,
            'error': self.y_test - self.predictions
        })
        
        # Show metrics
        print(f"MSE: {self.mse:.4f}")
        print(f"R²: {self.r2:.4f}")
        
        # Show a few examples
        print(f"Sample predictions:\n{results.head()}")

if __name__ == '__main__':
    ScoringFlow()