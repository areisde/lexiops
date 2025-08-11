# app/llm.py
from __future__ import annotations
import os
from typing import Tuple
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_client_and_model() -> Tuple[object, str]:
    """
    Returns (client, model_identifier)

    Priority:
      1) Azure OpenAI if AZURE_OPENAI_ENDPOINT is set.
      2) OpenAI-compatible server if OPENAI_BASE_URL is set (vLLM/TGI/etc.).
      3) OpenAI default.

    Env vars:
      # Azure OpenAI
      AZURE_OPENAI_ENDPOINT=https://<your>.openai.azure.com/
      AZURE_OPENAI_API_KEY=...
      OPENAI_API_VERSION=2024-06-01
      AZURE_OPENAI_DEPLOYMENT=<deployment-name>

      # OpenAI-compatible (incl. OpenAI)
      OPENAI_API_KEY=...
      OPENAI_MODEL=gpt-4o-mini
      OPENAI_BASE_URL=http://localhost:8000/v1   # optional for self-hosted
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_version = os.getenv("OPENAI_API_VERSION", "2024-06-01")

    if azure_endpoint:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=60.0,
        )
        # For Azure, `model` is the *deployment name*
        model = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not model:
            raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is required for Azure OpenAI.")
        return client, model

    # OpenAI or any OpenAI-compatible server (vLLM/TGI) via base_url
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url or None,  # None -> default OpenAI endpoint
        timeout=60.0,
    )
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return client, model


def chat_complete(client, model: str, system: str, user: str,
                  temperature: float = 0.2, max_tokens: int | None = None) -> str:
    """
    Minimal chat completion wrapper. Returns the assistant text content.
    Works for OpenAI, Azure OpenAI, and OpenAI-compatible servers.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Defensive parsing across providers
    choice = resp.choices[0]
    content = getattr(choice.message, "content", None) or ""
    return content.strip()