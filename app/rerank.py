from __future__ import annotations
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Small but strong reranker; change if you like
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        pairs = [(query, h["text"]) for h in hits]
        enc = self.tokenizer.batch_encode_plus(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        scores = self.model(**enc).logits.squeeze(-1)
        scores = scores.detach().cpu().tolist()
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        hits.sort(key=lambda x: x["rerank_score"], reverse=True)
        return hits[:top_k]