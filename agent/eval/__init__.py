import mlflow
import mlflow.genai.datasets
import json
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

def create_dataset():
    # Set the experiment
    experiment_name = "lexiops-eval-dataset"
    DATABRICKS_PATH = str(Path(f"/Users/{os.environ['DATABRICKS_USERNAME_LOC']}/{experiment_name}"))
    mlflow.set_experiment(DATABRICKS_PATH)

    uc_schema = "workspace.default"
    evaluation_dataset_table_name = "lexiops_eval"

    eval_dataset = mlflow.genai.datasets.create_dataset(
        uc_table_name=f"{uc_schema}.{evaluation_dataset_table_name}",
    )

    with open("agent/eval/eval_dataset.json", "r") as f:
        eval_dataset_records = json.load(f)[:3]

    eval_dataset.merge_records(eval_dataset_records)

if __name__ == "__main__":
    create_dataset()