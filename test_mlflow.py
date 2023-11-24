import mlflow
from mlflow import log_metric, log_param, log_artifact
from random import choice

import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if __name__ == "__main__":
    # # Load the Iris dataset
    # X, y = datasets.load_iris(return_X_y=True)

    # # Split the data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    # # Define the model hyperparameters
    # params = {
    #     "solver": "lbfgs",
    #     "max_iter": 1000,
    #     "multi_class": "auto",
    #     "random_state": 8888,
    # }

    # # Train the model
    # lr = LogisticRegression(**params)
    # lr.fit(X_train, y_train)

    # # Predict on the test set
    # y_pred = lr.predict(X_test)

    # # Calculate metrics
    # accuracy = accuracy_score(y_test, y_pred)

    ###################################################

    # Set our tracking server uri for logging
    # alternatively set $MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # somehow doesn't work witho file://
    # mlflow.set_tracking_uri("file:///Users/greyxray/Downloads/mlflow-course/mlruns")

    # Create a new MLflow Experiment
    # or use export MLFLOW_EXPERIMENT_ID
    # after mlflow experiments create --experiment-name produce-metrics
    mlflow.set_experiment("MLflow Quickstart")  # global level folder

    log_param("a", 1)
    log_param("verbosity", "DEBUG")

    log_metric("timestamp", 1000)
    log_metric("TTC", 33)

    print("Current tracking URI:", mlflow.get_tracking_uri())

    log_artifact("test_produced_dataset.csv")

    metric_names = ["cpu", "ram", "disk"]

    percentages = [i for i in range(0, 100)]

    # Log random 40 metrics
    for i in range(40):
        # each metric is automatically associated with a timestamp indicating the time at which the metric was logged
        log_metric(choice(metric_names), choice(percentages))

    # # Start an MLflow run
    # with mlflow.start_run():
    #     # Log the hyperparameters
    #     mlflow.log_params(params)

    #     # Log the loss metric
    #     mlflow.log_metric("accuracy", accuracy)

    #     # Set a tag that we can use to remind ourselves what this run was for
    #     mlflow.set_tag("Training Info", "Basic LR model for iris data")

    #     # Infer the model signature
    #     signature = infer_signature(X_train, lr.predict(X_train))

    #     # Log the model
    #     model_info = mlflow.sklearn.log_model(
    #         sk_model=lr,
    #         artifact_path="iris_model",
    #         signature=signature,
    #         input_example=X_train,
    #         registered_model_name="tracking-quickstart",
    #     )

    ##################################################
    # mlflow.set_tracking_uri("http://127.0.0.1:8080")
    # print("Current tracking URI:", mlflow.get_tracking_uri())
    # log_artifact("test_produced_dataset.csv")
    # # with mlflow.start_run():
    #     mlflow.log_artifact(
    #         "featest_produced_datasettures.csv",
    #         # artifact_path="features"
    #     )

    # # Create a features.txt artifact file
    # features = "rooms, zipcode, median_price, school_rating, transport"
    # with open("features.txt", "w") as f:
    #     f.write(features)

    # # With artifact_path=None write features.txt under
    # # root artifact_uri/artifacts directory
    # mlflow.end_run()
    # with mlflow.start_run():
    #     mlflow.log_artifact("features.txt")
    # mlflow.end_run()
