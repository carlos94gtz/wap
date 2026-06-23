#!/usr/bin/env python3
import json
import os
import socket
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

API_BASE = "https://api.openai.com/v1"
OPENAI_BETA_HEADER = "assistants=v2"


def api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in the environment.")
    key = key.strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is empty after trimming whitespace.")
    return key


def can_resolve_openai_host() -> bool:
    try:
        socket.getaddrinfo("api.openai.com", 443)
        return True
    except OSError:
        return False


def curl_json(
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, object]] = None,
    form: Optional[List[Tuple[str, str]]] = None,
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


def run_with_curl(file_path: str, question: str, model: str, vector_store_name: str) -> int:
    print("Using curl fallback because Python DNS resolution is failing in this environment.")

    uploaded_file = curl_json(
        "/files",
        method="POST",
        form=[
            ("purpose", "assistants"),
            ("file", f"@{file_path}"),
        ],
    )

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

    search = curl_json(
        f"/vector_stores/{vector_store['id']}/search",
        method="POST",
        payload={
            "query": question,
            "max_num_results": 5,
            "rewrite_query": True,
        },
        headers={"OpenAI-Beta": OPENAI_BETA_HEADER},
    )
    results = search.get("data") or []
    print(f"Vector search results: {len(results)}")
    for idx, result in enumerate(results[:3], start=1):
        content = result.get("content") or []
        text = content[0].get("text", "") if content else ""
        print(f"Result {idx}: score={result.get('score')} file={result.get('filename')}")
        if text:
            print(text[:500])

    response = curl_json(
        "/responses",
        method="POST",
        payload={
            "model": model,
            "input": question,
            "tools": [
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store["id"]],
                    "max_num_results": 5,
                }
            ],
            "include": ["file_search_call.results"],
        },
    )

    print(response.get("output_text") or json.dumps(response, ensure_ascii=False, indent=2))
    return 0


def run_with_sdk(file_path: str, question: str, model: str, vector_store_name: str) -> int:
    from openai import APIConnectionError, OpenAI, RateLimitError

    try:
        client = OpenAI()

        with open(file_path, "rb") as f:
            uploaded_file = client.files.create(
                file=f,
                purpose="assistants",
            )

        vector_store = client.vector_stores.create(name=vector_store_name)
        print(f"Vector store: {vector_store.id}")
        print(f"Uploaded file: {uploaded_file.id}")

        batch = client.vector_stores.file_batches.create_and_poll(
            vector_store_id=vector_store.id,
            file_ids=[uploaded_file.id],
        )
        print(f"Batch status: {batch.status}")

        search_results = client.vector_stores.search(
            vector_store.id,
            query=question,
            max_num_results=5,
            rewrite_query=True,
        )
        search_data = getattr(search_results, "data", [])
        print(f"Vector search results: {len(search_data)}")
        for idx, result in enumerate(search_data[:3], start=1):
            content = getattr(result, "content", [])
            text = content[0].text if content else ""
            print(f"Result {idx}: score={result.score} file={result.filename}")
            if text:
                print(text[:500])

        response = client.responses.create(
            model=model,
            input=question,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store.id],
                    "max_num_results": 5,
                }
            ],
            include=["file_search_call.results"],
        )

        output_text = getattr(response, "output_text", None)
        if output_text:
            print(output_text)
        else:
            print(response)

        return 0
    except APIConnectionError as exc:
        print(f"SDK connection failed, using curl fallback: {exc}")
        return run_with_curl(file_path, question, model, vector_store_name)
    except RateLimitError as exc:
        print(f"OpenAI quota exceeded: {exc}")
        print("Enable billing or add quota to the project, then rerun the same command.")
        return 1


def main() -> int:
    file_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPENAI_FILE_PATH")
    question = sys.argv[2] if len(sys.argv) > 2 else os.getenv("OPENAI_QUESTION", "Resume el documento")
    model = os.getenv("OPENAI_MODEL", "gpt-5.5")
    vector_store_name = os.getenv("OPENAI_VECTOR_STORE_NAME", "mi-rag")

    if not file_path:
        print("Usage: python work/openai_file_search_demo.py /ruta/al/archivo.pdf \"tu pregunta\"")
        return 1

    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY in the environment.")
        return 1

    file_path = file_path.strip().strip("'\"")
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return 1

    if can_resolve_openai_host():
        return run_with_sdk(file_path, question, model, vector_store_name)

    return run_with_curl(file_path, question, model, vector_store_name)


if __name__ == "__main__":
    raise SystemExit(main())
