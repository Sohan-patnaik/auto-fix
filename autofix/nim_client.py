"""Thin wrapper around NVIDIA AI Endpoints using LangChain."""

import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from pathlib import Path

# Path to project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from project root
load_dotenv(BASE_DIR / ".env")

LLM_MODEL = os.environ["NIM_LLM_MODEL"]
EMBED_MODEL = os.environ["NIM_EMBED_MODEL"]
API_KEY = os.environ["NVIDIA_API_KEY"]

# Initialize models
llm = ChatNVIDIA(
    model=LLM_MODEL,
    api_key=API_KEY,
)

embedder = NVIDIAEmbeddings(
    model=EMBED_MODEL,
    api_key=API_KEY,
)


def embed(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """
    Embed a list of texts.

    Args:
        texts: List of strings to embed.
        input_type: 'passage' for indexing or 'query' for search.
                    (Retained for API compatibility.)

    Returns:
        List of embedding vectors.
    """
    if input_type == "query":
        return [embedder.embed_query(text) for text in texts]
    return embedder.embed_documents(texts)


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    """
    Generate a chat completion.
    """
    response = llm.invoke(
        [
            ("system", system),
            ("human", user),
        ],
        temperature=temperature,
    )
    return response.content