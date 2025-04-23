from metaflow import FlowSpec, step, Parameter, conda_base, kubernetes, retry, timeout, catch, resources
import mlflow
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

MLFLOW_SERVER_URL = "https://mlflow-server-mvrsajmi4q-wl.a.run.app"

@conda_base(libraries={
    'numpy': '1.23.5',
    'scikit-learn': '1.2.2',
    'pandas': '2.0.0',
    'mlflow': '2.15.1'
}, python='3.9.16')
class TrainingFlowGCP(FlowSpec):

    random_state = Parameter('random_state', default=42, type=int)
    n_estimators = Parameter('n_estimators', default=100, type=int)
    max_depth = Parameter('max_depth', default=10, type=int)

    @step
    def start(self):
        housing = fetch_california_housing()
        X = pd.DataFrame(housing['data'], columns=housing['feature_names'])
        y = housing['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        self.max_depths = list(range(5, 20, 5))
        self.next(self.train_model, foreach='max_depths')

    @kubernetes
    @resources(cpu=1, memory=2000)
    @retry(times=3)
    @timeout(minutes=10)
    @catch(var='run_error')
    @step
    def train_model(self):
        max_depth = self.input
        mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
        mlflow.set_experiment("housing_experiment_gcp")
        model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=max_depth, random_state=self.random_state, n_jobs=-1)
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        with mlflow.start_run(run_name=f"rf_max_depth_{max_depth}") as run:
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            if max_depth < 15:  # <- Skip large model upload
                mlflow.sklearn.log_model(model, "model")
            else:
                print("Skipping model logging to avoid 413 error (model too large)")

            self.model = model
            self.mse = mse
            self.r2 = r2
            self.used_max_depth = max_depth
            self.run_id = run.info.run_id

        self.next(self.join_models)

    @step
    def join_models(self, inputs):
        self.models = [inp.model for inp in inputs]
        self.mse_values = [inp.mse for inp in inputs]
        self.r2_values = [inp.r2 for inp in inputs]
        self.max_depths = [inp.used_max_depth for inp in inputs]
        self.run_ids = [inp.run_id for inp in inputs]
        best_idx = np.argmin(self.mse_values)
        self.best_model = self.models[best_idx]
        self.best_mse = self.mse_values[best_idx]
        self.best_r2 = self.r2_values[best_idx]
        self.best_max_depth = self.max_depths[best_idx]
        self.best_run_id = self.run_ids[best_idx]
        self.next(self.register_model)

    @step
    def register_model(self):
        mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
        model_uri = f"runs:/{self.best_run_id}/model"
        try:
            registered_model = mlflow.register_model(model_uri=model_uri, name="housing-model-gcp")
            print(f"Model registered as 'housing-model-gcp', version: {registered_model.version}")
        except Exception as e:
            print("Model too large to register, skipping registration.")
        self.next(self.end)

    @step
    def end(self):
        print("Flow completed.")
        print(f"Best model: max_depth={self.best_max_depth} | MSE: {self.best_mse:.4f} | R²: {self.best_r2:.4f}")
        print(f"Run ID: {self.best_run_id}")

if __name__ == '__main__':
    TrainingFlowGCP()
