from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List, Dict, Any

import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import SecretStr

import requests

# Load environment variables from .env file
load_dotenv()

API_URL = os.getenv("APP_URL", "http://127.0.0.1:8000/ask")

def load_evalset(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def create_azure_llm():
    """Create Azure OpenAI LLM for RAGAS evaluation"""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
    
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=SecretStr(api_key),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-06-01"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=0.0,
    )

def create_local_embeddings():
    """Create local HuggingFace embeddings for RAGAS evaluation (same as used for indexing)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def ask(q: str, k: int = 5, **kwargs) -> Dict[str, Any]:
    params = {"q": q, "k": k}
    params.update(kwargs)  # Add any additional parameters like 'pv'
    r = requests.get(API_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def main(eval_path: Path, k: int, prompt_version: str | None):
    items = load_evalset(eval_path)

    # Gather predictions + contexts
    questions, answers, contexts = [], [], []
    for row in items:
        params = {"q": row["question"], "k": k}
        if prompt_version:
            params["pv"] = prompt_version
        resp = ask(**params)
        questions.append(row["question"])
        answers.append(resp["answer"])
        # Ragas expects list[str] for contexts per Q
        ctx_texts = [hit["text"] for hit in resp["hits"]]
        contexts.append(ctx_texts)

    gt = [row["ground_truth"] for row in items]

    ds = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": gt,
    })

    mlflow.set_experiment("eval")
    with mlflow.start_run(run_name=f"ragas_eval_{prompt_version or 'default'}"):
        # Log basic parameters
        if prompt_version:
            mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("k", k)
        mlflow.log_param("eval_dataset_size", len(items))
        
        # Log the evaluation dataset as CSV
        eval_df = ds.to_pandas()
        eval_csv_path = Path("eval_dataset.csv")
        eval_df.to_csv(eval_csv_path, index=False)
        mlflow.log_artifact(str(eval_csv_path), artifact_path="datasets")
        eval_csv_path.unlink(missing_ok=True)

        # Create Azure OpenAI LLM and local embeddings for RAGAS evaluation
        azure_llm = create_azure_llm()
        local_embeddings = create_local_embeddings()

        # Log model info
        mlflow.log_param("llm_provider", "azure_openai")
        mlflow.log_param("llm_model", os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"))
        mlflow.log_param("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

        results = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=azure_llm,
            embeddings=local_embeddings,
        )
        scores = results.to_pandas().mean(numeric_only=True).to_dict()  # average scores

        # Log RAGAS metrics properly
        for metric_name, score in scores.items():
            mlflow.log_metric(f"ragas_{metric_name}", float(score))
        
        # Log overall quality score
        overall_score = sum(scores.values()) / len(scores)
        mlflow.log_metric("ragas_overall_score", overall_score)

        # Save per-question details
        results_df = results.to_pandas()
        out = Path("eval_results.json")
        out.write_text(results_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out), artifact_path="ragas")
        
        # Log results as CSV
        results_csv_path = Path("eval_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        mlflow.log_artifact(str(results_csv_path), artifact_path="results")
        results_csv_path.unlink(missing_ok=True)

        print("RAGAS scores:", scores)
        
        # Get current run info for model registration
        current_run = mlflow.active_run()
        if current_run and overall_score >= 0.7:  # Configurable threshold
            run_id = current_run.info.run_id
            model_name = f"lexiops-rag-eval-{prompt_version or 'default'}"
            model_info = {
                "prompt_version": prompt_version or "default",
                "k": k,
                "scores": scores,
                "overall_score": overall_score
            }
            
            # Log model metadata
            mlflow.log_dict(model_info, "model_metadata.json")
            
            # Register model
            model_uri = f"runs:/{run_id}/model_metadata.json"
            mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={
                    "type": "rag_evaluation",
                    "prompt_version": prompt_version or "default",
                    "overall_score": str(overall_score)
                }
            )
            print(f"Registered model: {model_name}")
        else:
            if current_run:
                print(f"Overall score {overall_score:.3f} below threshold, not registering model")
            else:
                print("No active MLflow run found")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", default="eval/evalset.jsonl", type=Path)
    ap.add_argument("--k", default=5, type=int)
    ap.add_argument("--prompt-version", default=None)
    args = ap.parse_args()
    main(args.eval_path, args.k, args.prompt_version)