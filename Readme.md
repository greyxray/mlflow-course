Based on [MLOps tools: MLflow and Huggingface](https://www.coursera.org/learn/mlops-mlflow-huggingface-duke)
# General UI notes
- filtering only looks on LAST metrics per experiments
- parameters specified in MLproject are logged automatically
    - each param is logged once per session. You can overwrite it but probably shouldn't

# Note
- see mlflow-demo for conda instal
- course link: https://www.coursera.org/learn/mlops-mlflow-huggingface-duke/lecture/2ihaD/connecting-mlflow-to-databricks


# What it can do
It will fetch the project, run it depending on MLproject and log the stuff
```sh
    # in case using no conda
    poetry shell
    export PATH="/usr/local/anaconda3/bin:$PATH"

    mlflow run git@github.com:databricks/mlflow-example.git -P alpha=5
    # or if cloned already:
    mlflow run mlflow-example -P alpha=5

    poetry run mlflow run mlflow-example -P alpha=5
```
