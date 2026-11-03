"""Weaviate vector store utilities for HOLO-GURU.

Uses self-provided embeddings so the application keeps full control over the
embedding model while Weaviate provides persistent vector + BM25 hybrid search.
"""
import logging
from functools import lru_cache
from typing import Iterable, List, Optional

import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def embed_texts(texts: Iterable[str], model_name: str) -> List[List[float]]:
    texts = list(texts)
    if not texts:
        return []
    model = get_embedder(model_name)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return vectors.tolist()


def connect(host: str = "localhost", http_port: int = 8080, grpc_port: int = 50051):
    client = weaviate.connect_to_local(
        host=host,
        port=http_port,
        grpc_port=grpc_port,
    )
    if not client.is_ready():
        client.close()
        raise RuntimeError("Weaviate is not ready")
    return client


def ensure_collection(client, collection_name: str):
    if not client.collections.exists(collection_name):
        client.collections.create(
            name=collection_name,
            vector_config=wvc.config.Configure.Vectors.self_provided(),
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="book_title", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="book_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="page", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="section", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="file_path", data_type=wvc.config.DataType.TEXT),
            ],
        )
        logger.info("Created Weaviate collection '%s'", collection_name)
    return client.collections.use(collection_name)


def make_book_filter(book_ids: Optional[List[str]]):
    if not book_ids:
        return None
    if len(book_ids) == 1:
        return wvc.query.Filter.by_property("book_id").equal(book_ids[0])
    return wvc.query.Filter.any_of([
        wvc.query.Filter.by_property("book_id").equal(book_id)
        for book_id in book_ids
    ])
