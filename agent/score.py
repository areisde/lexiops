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
from typing import Dict, Any, List
import json
from .run import stream_graph_updates as predict_fn
from mlflow.utils.git_utils import get_git_commit
from pathlib import Path
import yaml


# ===== EVAL DATASET =====
with open("agent/eval/eval_dataset.json", "r") as f:
    eval_dataset = json.load(f)[:3]

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

def evaluate_legal_rag_agent(data, predict_fn):
    """
    Evaluate the legal RAG agent with MLflow's built-in scorers.
    
    Args:
        data: Evaluation dataset
        predict_fn: Function that generates responses
        extra_scorers: Additional custom scorers to include
        
    Returns:
        MLflow evaluation results
    """
    #mlflow.set_experiment("lexiops-eval")

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
    with open(CONFIG_PATH, "r") as f:
        model_params = yaml.safe_load(f)

    # Set active model context - all traces will link to this version
    model_params = {
        "provider" : model_params["model"]["id"].split(":")[0],
        "model" : model_params["model"]["id"].split(":")[1],
        "tools" : model_params["tools"]["enabled"],
    }

    with mlflow.set_active_model(name=version_name) as active_model:
        model_id = active_model.model_id
        mlflow.log_model_params(model_id=model_id, params=model_params)

        # Enable automatic tracing
        mlflow.langchain.autolog()

        # Run evaluation
        mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=legal_rag_scorers,
        )

if __name__ == "__main__":
    # Example usage
    evaluate_legal_rag_agent(eval_dataset, predict_fn)
