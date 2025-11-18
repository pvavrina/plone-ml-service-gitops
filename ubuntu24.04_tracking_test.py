# Load environment variables from .env file
#from dotenv import load_dotenv
#load_dotenv()

# MLflow Tracking Test Script using Scikit-learn
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np

# 1. Configuration of the Tracking Server
# Set the tracking URI to the stable MLflow FQDN address.
MLFLOW_TRACKING_URI = "http://mlf.club-linux.ch/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Define the experiment name (will be created if it doesn't exist)
mlflow.set_experiment("Iris_Classification_Test")

# 2. Data Preparation
iris = load_iris()
X = iris.data[:, :2] # Use the first two features
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Start the MLflow Run
with mlflow.start_run():
    # Define model hyperparameters.
    C_param = 0.1
    max_iter_param = 1000

    # Log parameters to MLflow.
    mlflow.log_param("C", C_param)
    mlflow.log_param("max_iter", max_iter_param)
    
    # 4. Model Training
    model = LogisticRegression(C=C_param, max_iter=max_iter_param)
    model.fit(X_train, y_train)
    
    # 5. Evaluation and Metric Logging
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log the final metric (Accuracy).
    mlflow.log_metric("accuracy", accuracy)
    
    # 6. Model Registration (Artifact)
    # Log the scikit-learn model as an artifact.
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="IrisLogRegModel"
    )

    print(f"\n--- MLflow Result ---")
    print(f"Model trained with Accuracy: {accuracy:.4f}")
    print(f"Tracking URI used: {mlflow.get_tracking_uri()}")
    print(f"Experiment Name: {mlflow.get_experiment(mlflow.active_run().info.experiment_id).name}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("\nPlease check the results on http://mlf.club-linux.ch/")
