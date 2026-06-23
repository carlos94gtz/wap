#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

API_BASE = "https://api.openai.com/v1"
OPENAI_BETA_HEADER = "assistants=v2"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
DEFAULT_OCR_MODEL = os.getenv("OPENAI_OCR_MODEL", DEFAULT_MODEL)
DEFAULT_VECTOR_STORE_NAME = os.getenv("OPENAI_VECTOR_STORE_NAME", "mi-rag")
DEFAULT_OCR_PROMPT = (
    "Transcribe all readable text from this document. Return plain text only, "
    "preserve headings and line breaks when possible, and omit commentary."
)
CACHE_PATH = Path(os.getenv("OPENAI_RAG_CACHE", Path(__file__).with_name(".rag_cache.json")))


def api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in the environment.")
    key = key.strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is empty after trimming whitespace.")
    return key


def curl_json(
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, object]] = None,
    form: Optional[List[tuple[str, str]]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    cmd = [
        "curl",
        "-sS",
        "--fail-with-body",
        "-X",
        method,
        f"{API_BASE}{path}",
        "-H",
        f"Authorization: Bearer {api_key()}",
    ]

    for name, value in (headers or {}).items():
        cmd.extend(["-H", f"{name}: {value}"])

    if payload is not None:
        cmd.extend(["-H", "Content-Type: application/json"])
        cmd.extend(["--data-raw", json.dumps(payload)])

    if form is not None:
        for field, value in form:
            cmd.extend(["-F", f"{field}={value}"])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or f"curl exited with code {result.returncode}"
        raise RuntimeError(details)

    output = result.stdout.strip()
    return json.loads(output) if output else {}


def load_cache() -> Dict[str, Dict[str, str]]:
    if not CACHE_PATH.is_file():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(cache: Dict[str, Dict[str, str]]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_key(path: Path, ocr_model: str) -> str:
    material = f"{file_sha256(path)}::{ocr_model}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def extract_response_text(payload: Dict[str, object]) -> str:
    texts: List[str] = []

    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()

    output = payload.get("output") or []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content") or []
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())

    return "\n".join(texts).strip()


def extract_search_snippets(payload: Dict[str, object]) -> List[Dict[str, str]]:
    results = payload.get("data") or []
    snippets: List[Dict[str, str]] = []
    if not isinstance(results, list):
        return snippets

    for item in results:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        text = ""
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                text = first.get("text", "") or ""
        snippets.append(
            {
                "score": str(item.get("score", "")),
                "filename": str(item.get("filename", "")),
                "text": text,
            }
        )
    return snippets


def openai_upload_file(path: Path) -> Dict[str, object]:
    return curl_json(
        "/files",
        method="POST",
        form=[
            ("purpose", "assistants"),
            ("file", f"@{path}"),
        ],
    )


def openai_ocr_pdf(pdf_path: Path, ocr_model: str) -> str:
    uploaded = openai_upload_file(pdf_path)
    response = curl_json(
        "/responses",
        method="POST",
        payload={
            "model": ocr_model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": uploaded["id"],
                            "detail": "high",
                        },
                        {
                            "type": "input_text",
                            "text": DEFAULT_OCR_PROMPT,
                        },
                    ],
                }
            ],
            "max_output_tokens": 8000,
        },
    )
    text = extract_response_text(response)
    if not text:
        raise RuntimeError("OCR returned no text.")
    return text


def ensure_prepared_text(source_path: Path, ocr_model: str) -> Path:
    if not is_pdf(source_path):
        return source_path

    ocr_path = source_path.with_suffix(".ocr.txt")
    if ocr_path.is_file() and ocr_path.stat().st_mtime >= source_path.stat().st_mtime and ocr_path.stat().st_size > 0:
        return ocr_path

    print("Running OCR...")
    text = openai_ocr_pdf(source_path, ocr_model)
    ocr_path.write_text(text, encoding="utf-8")
    print(f"OCR saved to: {ocr_path}")
    return ocr_path


def vector_store_exists(vector_store_id: str) -> bool:
    try:
        curl_json(
            f"/vector_stores/{vector_store_id}",
            headers={"OpenAI-Beta": OPENAI_BETA_HEADER},
        )
        return True
    except RuntimeError:
        return False


def prepare_vector_store(source_path: Path, prepared_path: Path, vector_store_name: str, ocr_model: str) -> str:
    cache = load_cache()
    key = source_key(source_path, ocr_model)
    cached = cache.get(key) or {}
    vector_store_id = cached.get("vector_store_id", "")

    if vector_store_id and vector_store_exists(vector_store_id):
        print(f"Reusing vector store: {vector_store_id}")
        return vector_store_id

    uploaded_file = openai_upload_file(prepared_path)
    vector_store = curl_json(
        "/vector_stores",
        method="POST",
        payload={"name": vector_store_name},
        headers={"OpenAI-Beta": OPENAI_BETA_HEADER},
    )

    print(f"Vector store: {vector_store['id']}")
    print(f"Uploaded file: {uploaded_file['id']}")

    batch = curl_json(
        f"/vector_stores/{vector_store['id']}/file_batches",
        method="POST",
        payload={"file_ids": [uploaded_file["id"]]},
        headers={"OpenAI-Beta": OPENAI_BETA_HEADER},
    )

    batch_id = batch["id"]
    while True:
        batch = curl_json(
            f"/vector_stores/{vector_store['id']}/file_batches/{batch_id}",
            headers={"OpenAI-Beta": OPENAI_BETA_HEADER},
        )
        status = batch.get("status")
        print(f"Batch status: {status}")
        if status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(2)

    if batch.get("status") != "completed":
        raise RuntimeError(f"Vector store batch finished with status: {batch.get('status')}")

    cache[key] = {
        "vector_store_id": vector_store["id"],
        "source_path": str(source_path),
        "prepared_path": str(prepared_path),
    }
    save_cache(cache)
    return vector_store["id"]


def ask_question(vector_store_id: str, question: str, model: str) -> None:
    search = curl_json(
        f"/vector_stores/{vector_store_id}/search",
        method="POST",
        payload={
            "query": question,
            "max_num_results": 5,
            "rewrite_query": True,
        },
        headers={"OpenAI-Beta": OPENAI_BETA_HEADER},
    )
    snippets = extract_search_snippets(search)
    print(f"Vector search results: {len(snippets)}")
    for idx, snippet in enumerate(snippets[:3], start=1):
        print(f"Result {idx}: score={snippet['score']} file={snippet['filename']}")
        if snippet["text"]:
            print(snippet["text"][:500])

    response = curl_json(
        "/responses",
        method="POST",
        payload={
            "model": model,
            "input": question,
            "tools": [
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                    "max_num_results": 5,
                }
            ],
            "include": ["file_search_call.results"],
        },
    )

    answer = extract_response_text(response)
    if answer:
        print(answer)
        return

    print(json.dumps(response, ensure_ascii=False, indent=2))


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python work/rag_ocr_reuse.py "/ruta/al/archivo.pdf" "Pregunta 1" "Pregunta 2"')
        return 1

    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY in the environment.")
        return 1

    source_path = Path(sys.argv[1].strip().strip("'\"")).expanduser().resolve()
    if not source_path.is_file():
        print(f"File not found: {source_path}")
        return 1

    questions = [arg.strip().strip("'\"") for arg in sys.argv[2:] if arg.strip()]
    if not questions:
        questions = [os.getenv("OPENAI_QUESTION", "Resume el documento")]

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    ocr_model = os.getenv("OPENAI_OCR_MODEL", DEFAULT_OCR_MODEL)
    vector_store_name = os.getenv("OPENAI_VECTOR_STORE_NAME", DEFAULT_VECTOR_STORE_NAME)

    prepared_path = ensure_prepared_text(source_path, ocr_model)
    vector_store_id = prepare_vector_store(source_path, prepared_path, vector_store_name, ocr_model)
    print(f"Using vector store: {vector_store_id}")

    for index, question in enumerate(questions, start=1):
        if len(questions) > 1:
            print(f"\nQ{index}: {question}")
        ask_question(vector_store_id, question, model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
