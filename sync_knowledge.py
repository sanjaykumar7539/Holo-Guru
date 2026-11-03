"""Incrementally ingest new/changed knowledge files without deleting old data."""
import hashlib
import json
import logging
from pathlib import Path

import config
from core.ingestion import ingest_book
from core.vector_store import connect, ensure_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("knowledge-sync")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    state_path = Path(config.SYNC_STATE_FILE)
    try:
        state = json.loads(state_path.read_text("utf-8")) if state_path.exists() else {}
    except Exception:
        state = {}

    client = connect(config.WEAVIATE_HOST, config.WEAVIATE_HTTP_PORT, config.WEAVIATE_GRPC_PORT)
    collection = ensure_collection(client, config.COLLECTION_NAME)
    updated = skipped = failed = 0
    next_state = dict(state)
    try:
        for path in sorted(Path(config.KNOWLEDGE_DIR).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
                continue
            key = str(path.relative_to(config.KNOWLEDGE_DIR))
            try:
                current = digest(path)
                if state.get(key) == current:
                    skipped += 1
                    continue
                info = ingest_book(str(path), collection, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                next_state[key] = current
                if info.get("duplicate"):
                    skipped += 1
                else:
                    updated += 1
            except Exception as exc:
                failed += 1
                log.error("Old knowledge preserved; failed to ingest %s: %s", key, exc)
        state_path.write_text(json.dumps(next_state, indent=2), encoding="utf-8")
    finally:
        client.close()
    log.info("Sync complete | updated=%d skipped=%d failed=%d", updated, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
