# Embeddings System

Verified against the current repository state on 2026-04-13.

The embeddings subsystem powers vector search, similarity detection, and cross-encoder reranking. It also provides the encoder used by prompt-intent classification. Most of the implementation lives in `src/ormah/embeddings/`.

## Architecture

```mermaid
graph TD
    subgraph "Public API"
        ENC["get_encoder()<br/>encoder.py"]
    end

    subgraph "Adapters (pluggable)"
        LOCAL["LocalAdapter<br/>FastEmbed / BGE<br/>CPU-only, ~420MB"]
        OLLAMA["OllamaAdapter<br/>HTTP to localhost:11434"]
        LITELLM["LiteLLMAdapter<br/>OpenAI, Gemini, Voyage, etc."]
    end

    subgraph "Storage"
        VS["VectorStore<br/>sqlite-vec virtual table"]
    end

    subgraph "Search"
        HS["HybridSearch<br/>FTS + Vector + RRF"]
    end

    subgraph "Precision"
        RERANK["Reranker<br/>Cross-encoder MiniLM"]
    end

    ENC --> LOCAL
    ENC --> OLLAMA
    ENC --> LITELLM

    LOCAL --> VS
    OLLAMA --> VS
    LITELLM --> VS

    VS --> HS
    HS --> RERANK

    style ENC fill:#74b3a5,color:#000
```

## Embedding Adapter Interface

**Code**: `embeddings/base.py`

```python
class EmbeddingAdapter(ABC):
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Single text → L2-normalized vector"""

    @abstractmethod
    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Batch encoding → normalized vectors"""

    def encode_query(self, text: str) -> np.ndarray:
        """Query-specific encoding (may add prefix)"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimensionality"""
```

## Local Adapter (Default)

**Code**: `embeddings/local_adapter.py`

- **Model**: BAAI/bge-base-en-v1.5
- **Dimensions**: 768
- **Library**: FastEmbed (ONNX runtime, CPU-only)
- **Size**: ~420MB download on first use
- **Cache**: `~/.local/share/ormah/models/`

```python
# Query encoding adds special prefix automatically
model.query_embed("what database does ormah use?")
# → internally: "Represent this sentence for searching relevant passages: what database does ormah use?"

# Document encoding (no prefix)
model.embed("Chose SQLite over Postgres for local-first design")
```

**Lazy loading**: Model is downloaded and loaded on first `encode()` call. Subsequent calls use a module-level `_model_cache` singleton keyed by model name.

### Local inference memory limits

Local embedding and reranker model loading and inference share one process-wide
worker thread (`embeddings/runtime.py`). This prevents concurrent requests from
multiplying activation memory or loading duplicate models, and reuses the same
native allocation context across calls. FastEmbed's lazy output iterators are
consumed on that worker. Nested calls, such as model loading during an encode,
execute directly on the worker to avoid deadlock.

Embedding and reranker inference batches contain at most eight inputs. Every
input is still processed in order, and query text remains intact for each
model's tokenizer to truncate at its supported token window. The default BGE
and MS MARCO models already cap sequences at 512 tokens; a character cutoff
would discard context earlier and could change retrieval meaning.

Both local models receive FastEmbed's top-level
`enable_cpu_mem_arena=False` option. This requires FastEmbed 0.7.4 or later;
passing an ONNX `session_options` object to FastEmbed is silently ignored by
these model wrappers. Arena disabling and a shared worker substantially reduce
retained RSS for concurrent inference, but they do not impose a hard process
memory ceiling. ORT and the system allocator can still retain memory, and
callers queue behind long local inference work. Remote embedding providers keep
their existing execution behavior.

Issue [#322](https://github.com/r-spade/ormah/issues/322) can be reproduced in
a fresh process against cached real models and an isolated temporary store:

```bash
PYTHONPATH=src python scripts/diag/local_inference_memory.py \
  --cache-dir /path/to/model-cache --nodes 45 --concurrency 2 --rounds 1 \
  --rss-cap-mib 2800
```

The related startup re-embedding interruption described in #322 is separate:
the current rebuild computes all vectors before persisting chunks. These local
inference limits do not add restart checkpoints to that rebuild.

### Why BGE?

- Runs entirely on CPU (no GPU needed)
- 768 dimensions gives good precision without excessive storage
- Asymmetric retrieval: different encoding for queries vs documents improves recall
- ~420MB is acceptable for a local-first tool

## Ollama Adapter

**Code**: `embeddings/ollama_adapter.py`

For users running Ollama locally:

```python
# HTTP POST to http://localhost:11434/api/embed
response = httpx.post(f"{base_url}/api/embed", json={
    "model": "nomic-embed-text",  # default
    "input": [text]
})
```

- Batch processing with 32-item chunks
- Auto-normalization of returned vectors

## LiteLLM Adapter

**Code**: `embeddings/litellm_adapter.py`

For cloud-hosted embeddings (OpenAI, Gemini, Voyage, Mistral):

```python
# Uses litellm's unified interface
response = litellm.embedding(model="text-embedding-3-small", input=[text])
```

- Batch processing with 32-item chunks
- Auto-normalization

## Encoder Factory

**Code**: `embeddings/encoder.py`, `embeddings/__init__.py`

```python
def get_encoder(settings=None) -> EmbeddingAdapter:
    if settings is None:
        settings = default_settings
    return get_adapter(settings)

def get_adapter(settings) -> EmbeddingAdapter:
    if settings.embedding_provider == "local":
        return LocalAdapter(model_name=settings.embedding_model)
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingAdapter(
            model=settings.embedding_model,
            base_url=settings.llm_base_url,
            dim=settings.embedding_dim,
        )
    if settings.embedding_provider == "litellm":
        return LiteLLMEmbeddingAdapter(
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
```

The current implementation caches one adapter per settings-object identity in a module-level `_adapter_cache`.

## Vector Store

**Code**: `embeddings/vector_store.py`

Wraps the `sqlite-vec` extension for vector storage and KNN search:

The backing store is the `node_vectors` sqlite-vec virtual table in the derived SQLite index.

```python
class VectorStore:
    def upsert(self, node_id: str, embedding: np.ndarray):
        """DELETE + INSERT in single transaction"""
        # Atomic: prevents reader seeing missing row

    def search(self, query_vec: np.ndarray, k: int) -> list[dict]:
        """KNN via sqlite-vec MATCH operator"""
        # SELECT id, distance FROM node_vectors
        # WHERE embedding MATCH ? AND k = ?
        # Returns L2 distance, converted:
        # cosine_similarity = 1 - (distance² / 2)

    def upsert_batch(self, items: list[tuple[str, np.ndarray]]):
        """Batch upsert in single transaction"""
```

### L2 to Cosine Conversion

sqlite-vec uses L2 (Euclidean) distance internally. For L2-normalized vectors (unit vectors), cosine similarity can be derived:

```
cosine_similarity = 1 - (L2_distance² / 2)
```

This works because for unit vectors: `||a - b||² = 2 - 2·cos(a,b)`, so `cos(a,b) = 1 - ||a-b||²/2`.

## Cross-Encoder Reranker

**Code**: `embeddings/reranker.py`

A precision filter used exclusively in the whisper pipeline. Unlike embeddings (which encode query and document separately), the cross-encoder sees both together.

### How It Works

```mermaid
flowchart LR
    subgraph "Bi-Encoder (fast, recall)"
        Q1[Query] --> ENC1[Encoder]
        D1[Document] --> ENC2[Encoder]
        ENC1 --> SIM["Cosine similarity<br/>score: 0.72"]
        ENC2 --> SIM
    end

    subgraph "Cross-Encoder (slow, precision)"
        Q2[Query] --> CE["Cross-Encoder<br/>sees both together"]
        D2[Document] --> CE
        CE --> SCORE["Relevance score<br/>score: +2.3"]
    end
```

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Input: (query, document) pairs
- Output: raw score in [-12, +6] range
- Positive = relevant, negative = irrelevant

### Linear Rescale

```python
def _linear_rescale(ce_score: float) -> float:
    """Maps [-12, +6] → [0, 1] linearly"""
    return max(0.0, min(1.0, (ce_score - (-12)) / (6 - (-12))))
```

Linear rescale (not sigmoid) is used because it preserves the cross-encoder's ability to **strongly suppress** irrelevant results. Sigmoid would flatten the extremes, reducing discrimination.

### Blending

```python
final_score = alpha * linear_rescale(ce_score) + (1 - alpha) * embedding_score
# alpha = 0.6 (60% cross-encoder weight)
```

## Model Cache

**Code**: `embeddings/cache.py`

```python
def get_fastembed_cache_dir() -> Path:
    """~/.local/share/ormah/models/ or FASTEMBED_CACHE_PATH env"""

def is_model_cached(model_name: str) -> bool:
    """Check if model directory exists"""
```

During `ormah setup`, both the embedding model and reranker model are preloaded so the first whisper call doesn't have a cold-start delay.

## Content Truncation

Before embedding, content is truncated to `embedding_max_content_chars` (default: 512 characters). This prevents long documents from getting "averaged out" embeddings that match everything weakly.

```python
text_to_embed = f"{title}\n{content[:512]}" if title else content[:512]
```

## Walkthrough: Encoding and Searching

Say we store: "Chose SQLite over Postgres because local-first doesn't need a server database"

1. **Encoding**: `LocalAdapter.encode("Chose SQLite over Postgres...")` → 768-dim normalized vector
2. **Storage**: `VectorStore.upsert(node_id, vector)` → INSERT into sqlite-vec
3. **Later, search**: User asks "what database does ormah use?"
4. **Query encoding**: `LocalAdapter.encode_query("what database does ormah use?")` → 768-dim query vector (with "Represent this sentence..." prefix)
5. **KNN search**: `VectorStore.search(query_vec, k=30)` → returns L2 distances
6. **Convert**: `cosine_sim = 1 - (distance² / 2)` → 0.82 for "Chose SQLite..."
7. **Threshold**: 0.82 > 0.4 → passes
8. **Feed into HybridSearch**: Combined with FTS5 results via RRF (see [03 - Search and Ranking](<./03 - Search and Ranking.md>))

## Related Docs

- [03 - Search and Ranking](<./03 - Search and Ranking.md>)
- [04 - Whisper - Involuntary Recall](<./04 - Whisper - Involuntary Recall.md>)
- [12 - Configuration Reference](<./12 - Configuration Reference.md>)
