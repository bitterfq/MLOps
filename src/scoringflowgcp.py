from metaflow import FlowSpec, step, Parameter, Flow, conda_base, kubernetes, retry, timeout, catch
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Your MLFlow server URL from the setup
MLFLOW_SERVER_URL = "https://mlflow-server-mvrsajmi4q-wl.a.run.app"

@conda_base(libraries={'numpy': '1.23.5', 'scikit-learn': '1.2.2', 'pandas': '2.0.0', 
                       'mlflow': '2.15.1'}, python='3.9.16')
class ScoringFlowGCP(FlowSpec):
    """
    Uses the model trained in TrainingFlowGCP to make predictions.
    """
    
    # Parameter we can change at runtime
    model_name = Parameter('model_name', default='housing-model-gcp', 
                          help='Name of the model in MLflow', type=str)
    model_stage = Parameter('model_stage', default='None', help='Model stage (None, Staging, Production)')
    
    @step
    def start(self):
        """
        Starts the flow by getting test data.
        """
        print("Starting scoring flow")
        
        # Get the test data from the latest training run
        train_flow = Flow('TrainingFlowGCP').latest_run
        self.X_test = train_flow['start'].task.data.X_test
        self.y_test = train_flow['start'].task.data.y_test
        self.scaler = train_flow['start'].task.data.scaler
        
        print(f"Test data loaded: {self.X_test.shape[0]} samples")
        
        self.next(self.load_model)
    
    @kubernetes
    @retry(times=3)
    @timeout(minutes=5)
    @catch(var='run_error')
    @step
    def load_model(self):
        """
        Loads the model from MLflow in GCP.
        """
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
        
        print(f"Loading model: {self.model_name}")
        
        # Load model by name and stage
        stage = None if self.model_stage == 'None' else self.model_stage
        model_uri = f"models:/{self.model_name}/{stage}"
        
        print(f"Loading model from {model_uri}")
        self.model = mlflow.pyfunc.load_model(model_uri)
        
        print("Model loaded successfully")
        
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
    ScoringFlowGCP()