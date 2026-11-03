# HOLO-GURU RAG migration

The old ChromaDB + MiniLM retrieval path has been replaced with:

- Weaviate as the persistent database/search engine.
- BAAI/bge-m3 embeddings generated locally by the app.
- Native Weaviate hybrid search: BM25 keyword + dense vector retrieval.
- Automatic query-aware hybrid weighting.
- BAAI/bge-reranker-v2-m3 cross-encoder reranking of retrieved candidates.
- Source/page/section metadata and stronger grounded-context construction.
- Duplicate chunk removal and content-based book IDs during ingestion.

## Run

Docker/GPU host:

```bash
docker compose up -d --build
```

CPU-only host: remove the NVIDIA `deploy` reservation block from
`docker-compose.yml`, then run the same command.

If running the Python app directly instead of in Docker, start only Weaviate:

```bash
docker compose up -d weaviate
pip install -r requirements.txt
python main.py
```

The local app connects to `localhost:8080` / gRPC `50051` by default.
Environment variables `WEAVIATE_HOST`, `WEAVIATE_HTTP_PORT`, and
`WEAVIATE_GRPC_PORT` can override these values.

## Note about GraphRAG

This migration intentionally does not label normal object references as full
GraphRAG. Weaviate is used here as the all-in-one hybrid retrieval database.
A later true graph-reasoning layer can be added if entity/relation extraction
and graph traversal are required.
