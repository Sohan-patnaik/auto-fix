"""Clone a repo and index its source files into a local Chroma collection."""
import os
import shutil
import chromadb
from git import Repo
import stat
from . import nim_client

CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".rb", ".rs", ".c", ".cpp", ".h"}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build"}
CHUNK_LINES = 80  # simple fixed-size chunking; good enough for MVP
CHUNK_OVERLAP = 10

def remove_readonly(func, path, exc_info):
    """Clear the read-only bit and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clone_repo(repo_url: str, dest: str) -> str:
    if os.path.exists(dest):
        shutil.rmtree(dest, onerror=remove_readonly)

    Repo.clone_from(repo_url, dest, depth=1)
    return dest


def _iter_source_files(repo_path: str):
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if os.path.splitext(f)[1] in CODE_EXTS:
                yield os.path.join(root, f)


def _chunk_file(path: str, repo_path: str):
    with open(path, "r", errors="ignore") as fh:
        lines = fh.readlines()
    rel = os.path.relpath(path, repo_path)
    chunks = []
    step = CHUNK_LINES - CHUNK_OVERLAP
    for start in range(0, max(len(lines), 1), step):
        block = lines[start:start + CHUNK_LINES]
        if not block:
            continue
        text = "".join(block).strip()
        if text:
            chunks.append({
                "text": text,
                "filepath": rel,
                "start_line": start + 1,
                "end_line": start + len(block),
            })
        if start + CHUNK_LINES >= len(lines):
            break
    return chunks


def build_index(repo_path: str, collection_name: str, chroma_path: str = "./chroma_db"):
    """Chunk every source file, embed via NIM, upsert into a fresh Chroma collection."""
    db = chromadb.PersistentClient(path=chroma_path)
    try:
        db.delete_collection(collection_name)
    except Exception:
        pass
    collection = db.create_collection(collection_name)

    all_chunks = []
    for filepath in _iter_source_files(repo_path):
        all_chunks.extend(_chunk_file(filepath, repo_path))

    if not all_chunks:
        raise RuntimeError("No source files found to index.")

    # Batch embed to keep request sizes reasonable.
    BATCH = 32
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i + BATCH]
        vectors = nim_client.embed([c["text"] for c in batch], input_type="passage")
        collection.add(
            ids=[f"{c['filepath']}:{c['start_line']}-{c['end_line']}" for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[{"filepath": c["filepath"], "start_line": c["start_line"], "end_line": c["end_line"]} for c in batch],
        )

    return collection, len(all_chunks)


def query_index(collection, issue_text: str, k: int = 8):
    vector = nim_client.embed([issue_text], input_type="query")[0]
    results = collection.query(query_embeddings=[vector], n_results=k)
    hits = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        hits.append({"text": doc, **meta})
    return hits
