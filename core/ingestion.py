"""Parse documents -> clean -> structure-aware chunk -> embed -> Weaviate."""
import hashlib
import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import weaviate.classes as wvc

import config
from core.vector_store import embed_texts
from core.knowledge_graph import index_chunks as index_graph_chunks, delete_book as delete_graph_book

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_pdf(path: str) -> List[Dict]:
    import fitz
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = _clean_text(page.get_text("text"))
        if text:
            pages.append({"text": text, "page": i + 1})
    doc.close()
    return pages


def _parse_docx(path: str) -> List[Dict]:
    from docx import Document
    doc = Document(path)
    chunks, page, buf = [], 1, []
    for para in doc.paragraphs:
        t = _clean_text(para.text)
        if not t:
            continue
        buf.append(t)
        if len(buf) >= 15:
            chunks.append({"text": "\n".join(buf), "page": page})
            page += 1
            buf = []
    if buf:
        chunks.append({"text": "\n".join(buf), "page": page})
    return chunks


def _parse_epub(path: str) -> List[Dict]:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    book = epub.read_epub(path)
    pages, idx = [], 1
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = _clean_text(soup.get_text("\n", strip=True))
        if len(text) > 50:
            pages.append({"text": text, "page": idx})
            idx += 1
    return pages


def _parse_txt(path: str) -> List[Dict]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks = [_clean_text(b) for b in re.split(r"\n\s*\n", text)]
    blocks = [b for b in blocks if b]
    return [{"text": b, "page": i + 1} for i, b in enumerate(blocks)]


def _one_page(text: str) -> List[Dict]:
    text = _clean_text(text)
    return [{"text": text, "page": 1}] if text else []


def _parse_html(path: str) -> List[Dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(Path(path).read_text("utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return _one_page(soup.get_text("\n", strip=True))


def _parse_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as source:
        for row in csv.reader(source):
            rows.append(" | ".join(cell.strip() for cell in row))
    return _one_page("\n".join(rows))


def _parse_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as source:
        value = json.load(source)
    return _one_page(json.dumps(value, ensure_ascii=False, indent=2))


def _parse_xml(path: str) -> List[Dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(Path(path).read_text("utf-8", errors="ignore"), "xml")
    return _one_page(soup.get_text("\n", strip=True))


def _parse_pptx(path: str) -> List[Dict]:
    from pptx import Presentation
    presentation = Presentation(path)
    pages = []
    for index, slide in enumerate(presentation.slides, 1):
        text = "\n".join(shape.text for shape in slide.shapes if hasattr(shape, "text"))
        text = _clean_text(text)
        if text:
            pages.append({"text": text, "page": index})
    return pages


def _parse_xlsx(path: str) -> List[Dict]:
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=True)
    pages = []
    for index, sheet in enumerate(workbook.worksheets, 1):
        rows = [" | ".join("" if value is None else str(value) for value in row)
                for row in sheet.iter_rows(values_only=True)]
        text = _clean_text(f"Sheet: {sheet.title}\n" + "\n".join(rows))
        if text:
            pages.append({"text": text, "page": index})
    workbook.close()
    return pages


PARSERS = {
    ".pdf": _parse_pdf, ".docx": _parse_docx, ".epub": _parse_epub,
    ".txt": _parse_txt, ".html": _parse_html, ".htm": _parse_html,
    ".csv": _parse_csv, ".json": _parse_json, ".xml": _parse_xml,
    ".pptx": _parse_pptx, ".xlsx": _parse_xlsx,
}


def _guess_section(text: str) -> str:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        return ""
    first = lines[0][:120]
    # Keep a short heading-like first line as metadata; avoid entire sentences.
    if len(first) <= 90 and first.count(".") <= 1:
        return first
    return ""


def _chunk_pages(pages: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    )
    result = []
    seen_hashes = set()
    for pg in pages:
        for s in splitter.split_text(pg["text"]):
            s = _clean_text(s)
            if len(s) <= 60:
                continue
            digest = hashlib.sha1(s.lower().encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            result.append({"text": s, "page": pg["page"], "section": _guess_section(s)})
    return result


def ingest_book(file_path: str, collection, chunk_size: int, overlap: int) -> Dict:
    ext = Path(file_path).suffix.lower()
    if ext not in PARSERS:
        raise ValueError(f"Unsupported format: {ext}")

    book_title = Path(file_path).stem
    if len(book_title) > 20 and any(c.isdigit() for c in book_title):
        book_title = "Uploaded Book"

    # Stable per-file content identity avoids duplicate re-ingestion of the same file.
    file_bytes = Path(file_path).read_bytes()
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    book_id = content_hash[:16]

    existing = collection.query.fetch_objects(
        filters=wvc.query.Filter.by_property("book_id").equal(book_id),
        limit=1,
        return_properties=["book_id"],
    )
    if existing.objects:
        logger.info("Skipping duplicate content: %s", book_title)
        return {
            "title": book_title, "book_id": book_id, "content_hash": content_hash,
            "chunks": 0, "pages": 0, "duplicate": True,
        }

    logger.info("Parsing '%s' (%s)...", book_title, ext)
    pages = PARSERS[ext](file_path)
    chunks = _chunk_pages(pages, chunk_size, overlap) if pages else []

    if not chunks:
        return {"title": book_title, "book_id": book_id, "chunks": 0, "pages": len(pages)}

    # Remove old copy before re-importing the same content.
    collection.data.delete_many(
        where=wvc.query.Filter.by_property("book_id").equal(book_id)
    )

    vectors = embed_texts([c["text"] for c in chunks], config.EMBED_MODEL)
    objects = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        objects.append(wvc.data.DataObject(
            properties={
                "text": chunk["text"],
                "book_title": book_title,
                "book_id": book_id,
                "page": int(chunk["page"]),
                "chunk_index": i,
                "section": chunk.get("section", ""),
                "file_path": file_path,
            },
            vector=vector,
        ))

    result = collection.data.ingest(objects)
    if getattr(result, "errors", None):
        raise RuntimeError(f"Weaviate ingest errors: {result.errors[:3]}")

    graph_edges = 0
    try:
        graph_edges = index_graph_chunks(book_id, book_title, chunks)
    except Exception as exc:
        logger.warning("Knowledge graph indexing skipped: %s", exc)

    logger.info("Ingested '%s' - %d chunks stored in Weaviate; %d graph edges indexed.", book_title, len(chunks), graph_edges)
    return {
        "title": book_title, "book_id": book_id, "content_hash": content_hash,
        "chunks": len(chunks), "pages": len(pages), "graph_edges": graph_edges, "duplicate": False,
    }


def list_books(collection) -> List[Dict]:
    try:
        results = collection.query.fetch_objects(
            limit=10000,
            return_properties=["book_id", "book_title", "file_path"],
        )
        seen, books = set(), []
        for obj in results.objects:
            meta = obj.properties
            bid = meta.get("book_id")
            if bid and bid not in seen:
                seen.add(bid)
                books.append({
                    "book_id": bid,
                    "title": meta.get("book_title") or "Unknown",
                    "file_path": meta.get("file_path") or "",
                })
        return books
    except Exception as exc:
        logger.error("List books failed: %s", exc)
        return []


def delete_book(book_id: str, collection) -> int:
    # Fetch first so the API can keep returning a deleted chunk count.
    matches = collection.query.fetch_objects(
        filters=wvc.query.Filter.by_property("book_id").equal(book_id),
        limit=10000,
        return_properties=["book_id"],
    )
    count = len(matches.objects)
    if count:
        collection.data.delete_many(
            where=wvc.query.Filter.by_property("book_id").equal(book_id)
        )
    try:
        delete_graph_book(book_id)
    except Exception as exc:
        logger.warning("Knowledge graph cleanup skipped: %s", exc)
    return count
