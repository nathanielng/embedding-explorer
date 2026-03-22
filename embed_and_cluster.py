#!/usr/bin/env python3
"""
Step 1: Embed text chunks, reduce dimensions, cluster, and export to JSON.

Usage:
    python embed_and_cluster.py --input chunks.txt --output embeddings.json
    python embed_and_cluster.py --input chunks.txt --output embeddings.json --provider openai
    python embed_and_cluster.py --input chunks.txt --output embeddings.json --provider ollama
    python embed_and_cluster.py --input chunks.txt --output embeddings.json --provider ollama --ollama-model embeddinggemma
    python embed_and_cluster.py --input chunks.json --output embeddings.json  # JSON array of strings

Requires:
    uv pip install boto3 openai ollama umap-learn hdbscan numpy scikit-learn python-dotenv
"""

import argparse
import json
import logging
import os
import sys

import boto3
import numpy as np
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

load_dotenv()


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

def embed_bedrock(chunks: list[str]) -> np.ndarray:
    """Embed using Amazon Bedrock Titan Embeddings V2."""
    model_id = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
    region = os.getenv("AWS_REGION", "us-east-1")

    client = boto3.client("bedrock-runtime", region_name=region)
    vectors = []

    for i, chunk in enumerate(chunks):
        logging.info(f"Bedrock: embedding chunk {i + 1}/{len(chunks)}")
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps({"inputText": chunk}),
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(response["body"].read())
        vectors.append(body["embedding"])

    return np.array(vectors, dtype=np.float32)


def embed_openai(chunks: list[str]) -> np.ndarray:
    """Embed using OpenAI text-embedding-3-small."""
    from openai import OpenAI

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    logging.info(f"OpenAI: embedding {len(chunks)} chunks with {model}")
    response = client.embeddings.create(input=chunks, model=model)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def embed_ollama(chunks: list[str], model: str, batch_size: int) -> np.ndarray:
    """Embed using a local Ollama model (e.g. qwen3-embedding:0.6b, embeddinggemma)."""
    import ollama

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    client = ollama.Client(host=host)

    # Verify the model is available before starting
    try:
        client.show(model)
    except ollama.ResponseError:
        logging.error(
            f"Ollama: model '{model}' not found locally. "
            f"Run: ollama pull {model}"
        )
        sys.exit(1)

    logging.info(f"Ollama: embedding {len(chunks)} chunks with '{model}' "
                 f"(host={host}, batch_size={batch_size})")

    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        end = min(i + batch_size, len(chunks))
        logging.info(f"Ollama: chunks {i + 1}–{end}/{len(chunks)}")
        response = client.embed(model=model, input=batch)
        vectors.extend(response["embeddings"])

    return np.array(vectors, dtype=np.float32)


EMBEDDING_PROVIDERS = {
    "bedrock": embed_bedrock,
    "openai": embed_openai,
    "ollama": embed_ollama,
}


# ---------------------------------------------------------------------------
# Dimensionality reduction + clustering
# ---------------------------------------------------------------------------

def reduce_umap(vectors: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    import umap

    logging.info(f"UMAP: reducing {vectors.shape} → 2D (n_neighbors={n_neighbors}, min_dist={min_dist})")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(vectors)


def cluster_hdbscan(vectors: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    import hdbscan

    logging.info(f"HDBSCAN: clustering with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    clusterer.fit(vectors)
    labels = clusterer.labels_  # -1 = noise
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logging.info(f"HDBSCAN: found {n_clusters} clusters, {n_noise} noise points")
    return labels


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_chunks(path: str) -> list[str]:
    """Load chunks from a .txt (one per line) or .json (array of strings) file."""
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON input must be an array of strings")
            return [str(item) for item in data]
        else:
            return [line.strip() for line in f if line.strip()]


def build_output(chunks: list[str], xy: np.ndarray, labels: np.ndarray) -> list[dict]:
    records = []
    for i, (chunk, (x, y), label) in enumerate(zip(chunks, xy, labels)):
        records.append({
            "id": i,
            "text": chunk,
            "x": round(float(x), 6),
            "y": round(float(y), 6),
            "cluster": int(label),          # -1 = noise/unclustered
        })
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed, reduce, cluster text chunks → JSON")
    p.add_argument("--input",  required=True, help="Path to chunks (.txt or .json)")
    p.add_argument("--output", required=True, help="Path to write output JSON")
    p.add_argument("--provider", default="bedrock", choices=list(EMBEDDING_PROVIDERS),
                   help="Embedding provider (default: bedrock)")
    # Ollama-specific
    p.add_argument("--ollama-model", default="qwen3-embedding:0.6b",
                   help="Ollama model tag (default: qwen3-embedding:0.6b)")
    p.add_argument("--ollama-batch-size", type=int, default=32,
                   help="Chunks per Ollama embed call (default: 32)")
    # UMAP
    p.add_argument("--umap-neighbors", type=int, default=15,
                   help="UMAP n_neighbors — higher = more global structure (default: 15)")
    p.add_argument("--umap-min-dist", type=float, default=0.1,
                   help="UMAP min_dist — lower = tighter clusters (default: 0.1)")
    # HDBSCAN
    p.add_argument("--min-cluster-size", type=int, default=3,
                   help="HDBSCAN min_cluster_size (default: 3)")
    p.add_argument("--min-samples", type=int, default=2,
                   help="HDBSCAN min_samples (default: 2)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load chunks
    logging.info(f"Loading chunks from: {args.input}")
    chunks = load_chunks(args.input)
    logging.info(f"Loaded {len(chunks)} chunks")

    if len(chunks) < 4:
        logging.error("Need at least 4 chunks to cluster meaningfully")
        sys.exit(1)

    # Embed
    if args.provider == "ollama":
        vectors = embed_ollama(chunks, model=args.ollama_model, batch_size=args.ollama_batch_size)
    else:
        embed_fn = EMBEDDING_PROVIDERS[args.provider]
        vectors = embed_fn(chunks)
    logging.info(f"Embeddings shape: {vectors.shape}")

    # UMAP → 2D
    xy = reduce_umap(vectors, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)

    # HDBSCAN cluster labels
    labels = cluster_hdbscan(xy, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)

    # Build and write output
    records = build_output(chunks, xy, labels)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logging.info(f"Wrote {len(records)} records to: {args.output}")
    logging.info("Done — pass this JSON file to the visualiser.")


if __name__ == "__main__":
    main()
