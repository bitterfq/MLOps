from metaflow import FlowSpec, step, Parameter
import mlflow
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os

class TrainingFlow(FlowSpec):
    """Training flow for California Housing dataset"""
    
    # Parameters
    random_state = Parameter('random_state', default=42, help='Random seed', type=int)
    n_estimators = Parameter('n_estimators', default=100, help='Number of trees', type=int)
    max_depth = Parameter('max_depth', default=10, help='Max tree depth', type=int)
    
    @step
    def start(self):
        """Load and prepare the data"""
        print("Loading California Housing dataset")
        
        # Load dataset
        housing = fetch_california_housing()
        X = pd.DataFrame(housing['data'], columns=housing['feature_names'])
        y = housing['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save data for later steps
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.scaler = scaler
        
        print(f"Data prepared. Training set: {X_train.shape}, Test set: {X_test.shape}")
        self.next(self.train_model)
    
    @step
    def train_model(self):
        """Train a Random Forest model"""
        print(f"Training RandomForest with {self.n_estimators} trees, max_depth={self.max_depth}")
        
        # Create local directory for MLflow
        os.makedirs("mlruns", exist_ok=True)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate model
        y_pred = model.predict(self.X_test_scaled)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Store model and results
        self.model = model
        self.mse = mse
        self.r2 = r2
        
        # Log to MLflow locally without server
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("housing_experiment")
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("max_depth", self.max_depth)
            
            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Save run ID for later
            self.run_id = run.info.run_id
        
        # Register model
        mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/model",
            name="housing-model"
        )
        
        print(f"Model trained. MSE: {mse:.4f}, R²: {r2:.4f}")
        print(f"Model registered in MLflow as 'housing-model', run_id: {self.run_id}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """End the flow"""
        print("Training flow completed successfully!")
        print(f"Model metrics - MSE: {self.mse:.4f}, R²: {self.r2:.4f}")
        print(f"Model registered in MLflow with run_id: {self.run_id}")

if __name__ == '__main__':
    TrainingFlow()