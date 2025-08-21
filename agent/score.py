"""
MLflow scorers for evaluating the legal query RAG agent.

This module provides comprehensive evaluation metrics for assessing:
- Factual accuracy of responses
- Appropriate tool usage (RAG search and citations)
- Legal reasoning quality
- Source attribution and grounding
- Response completeness
"""

from mlflow.genai.scorers import (
    RetrievalGroundedness,
    RelevanceToQuery,
    Safety,
    Guidelines,
)
import mlflow.genai
from mlflow.data.pandas_dataset import from_pandas
from typing import Dict, Any, List
import json
from .run import run_graph as predict_fn
from mlflow.utils.git_utils import get_git_commit
from pathlib import Path
import yaml
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# ===== EVAL DATASET =====
def dataset(app_name, git_commit):
    version_name = f"eval_dataset_{app_name}_{git_commit}"
    uc_schema = "workspace.default"
    evaluation_dataset_table_name = version_name
    table_name = f"{uc_schema}.{evaluation_dataset_table_name}"

    try:
        # Try to retrieve the dataset
        eval_dataset = mlflow.genai.datasets.get_dataset(uc_table_name=table_name)
    except:
        # Create the dataset if it doesn't exist
        eval_dataset = mlflow.genai.datasets.create_dataset(
            uc_table_name=table_name,
        )

        with open("agent/eval/eval_dataset.json", "r") as f:
            eval_dataset_records = json.load(f)[:3]
        eval_dataset.merge_records(eval_dataset_records)

    return eval_dataset

# ===== LEGAL RAG SCORERS =====
# Simple, clean evaluation using MLflow's built-in scorers

legal_rag_scorers = [
    RetrievalGroundedness(),  # Checks if response is grounded in retrieved legal documents
    Guidelines(
        name="uses_appropriate_tools",
        guidelines="The agent must use the rag_search tool to retrieve relevant legal information when answering legal queries.",
    ),
    Guidelines(
        name="legal_terminology_usage", 
        guidelines="The response must use appropriate legal terminology and formal language.",
    ),
    Guidelines(
        name="specific_article_references",
        guidelines="When asked about specific articles of the EU AI Act, the response must explicitly reference the correct article numbers.",
    ),
    RelevanceToQuery(),       # Checks if response addresses the legal query
    Safety(),                 # Checks for harmful or inappropriate content
]


# ===== EVALUATION FUNCTION =====
# Simple evaluation function using the clean built-in scorers

def evaluate_legal_rag_agent(predict_fn):
    """
    Evaluate the legal RAG agent with MLflow's built-in scorers.
    
    Args:
        data: Evaluation dataset
        predict_fn: Function that generates responses
        extra_scorers: Additional custom scorers to include
        
    Returns:
        MLflow evaluation results
    """
    mlflow.set_experiment(f"/Users/{os.environ['DATABRICKS_USERNAME_LOC']}/lexiops-eval")

    git_commit = get_git_commit(".")
    if git_commit:
        git_commit = git_commit[:8]  # Use short hash
    else:
        git_commit = "local-dev"  # Fallback if not in git repo
    
    # Create version identifier
    app_name = "lexiops"
    version_name = f"{app_name}-{git_commit}"

    # Load current app version configs
    config_root_dir = Path(__file__).resolve().parent.parent
    CONFIG_PATH = config_root_dir / "config/agent/agent.yaml"
    DATA_PATH = config_root_dir / "data/LATEST"
    with open(CONFIG_PATH, "r") as f:
        model_params = yaml.safe_load(f)

    data_snapshot = open(DATA_PATH, "r").read()

    # Set active model context - all traces will link to this version
    model_params = {
        "provider" : model_params["model"]["id"].split(":")[0],
        "model" : model_params["model"]["id"].split(":")[1],
        "tools" : model_params["tools"]["enabled"],
        "data_snapshot" : data_snapshot
    }

    # Set up dataset
    eval_dataset = dataset(app_name, git_commit)

    with mlflow.set_active_model(name=version_name) as active_model:
        model_id = active_model.model_id
        mlflow.log_model_params(model_id=model_id, params=model_params)

        # Run evaluation
        mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=predict_fn,
            scorers=legal_rag_scorers,
        )

if __name__ == "__main__":
    # Example usage
    evaluate_legal_rag_agent(predict_fn)
