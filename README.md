# Embedding Cluster Explorer

Two-step workflow: Python embeds + clusters → HTML visualises.

## Setup

```bash
# Create venv and install deps
uv venv ~/.venv
source ~/.venv/bin/activate
uv pip install boto3 openai ollama umap-learn hdbscan numpy scikit-learn python-dotenv
```

## Environment variables (.env)

```env
# Bedrock (default)
AWS_REGION=ap-southeast-1
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# Or OpenAI
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Or Ollama (optional overrides — defaults work for local Ollama)
OLLAMA_HOST=http://localhost:11434
```

## Step 1 — Python: embed, reduce, cluster

```bash
# From a plain text file (one chunk per line)
python embed_and_cluster.py \
  --input chunks.txt \
  --output embeddings.json

# From a JSON array of strings
python embed_and_cluster.py \
  --input chunks.json \
  --output embeddings.json \
  --provider openai

# Ollama — qwen3-embedding:0.6b (recommended default)
python embed_and_cluster.py \
  --input chunks.txt \
  --output embeddings.json \
  --provider ollama

# Ollama — switch model (e.g. embeddinggemma, all-minilm)
python embed_and_cluster.py \
  --input chunks.txt \
  --output embeddings.json \
  --provider ollama \
  --ollama-model embeddinggemma

# Ollama — tune batch size (lower if you hit OOM, raise for speed)
python embed_and_cluster.py \
  --input chunks.txt \
  --output embeddings.json \
  --provider ollama \
  --ollama-model qwen3-embedding:0.6b \
  --ollama-batch-size 16

# Tuning UMAP/HDBSCAN options
python embed_and_cluster.py \
  --input chunks.txt \
  --output embeddings.json \
  --umap-neighbors 20 \      # higher = more global structure
  --umap-min-dist 0.05 \     # lower = tighter visual clusters
  --min-cluster-size 5 \     # minimum points to form a cluster
  --min-samples 3            # HDBSCAN min_samples
```

### Output format

```json
[
  { "id": 0, "text": "chunk text here", "x": 0.42, "y": -1.3, "cluster": 2 },
  { "id": 1, "text": "another chunk",  "x": 1.11, "y": 0.88, "cluster": 2 },
  { "id": 5, "text": "outlier chunk",  "x": -3.1, "y": 4.2,  "cluster": -1 }
]
```

`cluster: -1` means the point is noise (HDBSCAN couldn't assign it to a cluster).

## Ollama setup (for --provider ollama)

```bash
# Pull your chosen model first — the script will error early if it's missing
ollama pull qwen3-embedding:0.6b   # recommended: 8K context, strong multilingual
ollama pull embeddinggemma          # alternative: 2K context, very small footprint
ollama pull all-minilm              # legacy: 512 token limit, prototyping only
```

## Step 2 — Browser: interactive visualisation

Open `visualizer.html` in any modern browser — no server needed.

```bash
open visualizer.html          # macOS
xdg-open visualizer.html      # Linux
```

Then **drag-and-drop** `embeddings.json` onto the canvas, or click **Load JSON**.

### Features

| Interaction | What it does |
|---|---|
| Hover a dot | Tooltip with chunk preview |
| Click a dot | Full chunk text in detail pane |
| Click cluster in sidebar | Highlight + zoom to that cluster |
| Click again / background | Deselect |
| Search box | Highlight matching chunks |
| Show noise toggle | Hide/show unclustered points |
| Scroll/pinch | Zoom the scatter plot |
| Drag | Pan |

## Tuning tips

- **Too many noise points** → lower `--min-cluster-size` or `--min-samples`
- **Clusters too merged** → lower `--umap-min-dist` (e.g. `0.02`)
- **Clusters too separated** → raise `--umap-min-dist` (e.g. `0.2`)
- **Local vs global structure** → raise `--umap-neighbors` for more global
- **Small dataset (<50 chunks)** → set `--min-cluster-size 2 --min-samples 1`
