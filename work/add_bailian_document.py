#!/usr/bin/env python3
import hashlib
import json
import os
import sys
import time
import urllib.request

from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_bailian20231229.client import Client as BailianClient
from alibabacloud_bailian20231229 import models as bailian_models


def build_client() -> BailianClient:
    access_key_id = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")
    access_key_secret = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
    endpoint = os.getenv("ALIBABA_CLOUD_ENDPOINT", "bailian.cn-beijing.aliyuncs.com")
    if not access_key_id or not access_key_secret:
        raise RuntimeError("Missing ALIBABA_CLOUD_ACCESS_KEY_ID or ALIBABA_CLOUD_ACCESS_KEY_SECRET")

    config = open_api_models.Config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        endpoint=endpoint,
    )
    return BailianClient(config)


def dump(resp) -> None:
    body = resp.body
    if hasattr(body, "to_map"):
        print(json.dumps(body.to_map(), ensure_ascii=False, indent=2))
    else:
        print(body)


def response_ok(resp) -> bool:
    body = getattr(resp, "body", None)
    if body is None:
        return False
    return getattr(body, "success", False) in {True, "true", "True"}


def http_upload(url: str, method: str, headers, file_path: str) -> None:
    with open(file_path, "rb") as f:
        data = f.read()

    request = urllib.request.Request(url, data=data, method=method)
    if headers:
        if hasattr(headers, "to_map"):
            headers = headers.to_map()
        for key, value in dict(headers).items():
            if value is not None:
                request.add_header(key, str(value))

    with urllib.request.urlopen(request, timeout=120) as response:
        response.read()


def wait_for_job(client: BailianClient, workspace_id: str, index_id: str, job_id: str) -> None:
    for _ in range(60):
        status_resp = client.get_index_job_status(
            workspace_id,
            bailian_models.GetIndexJobStatusRequest(
                index_id=index_id,
                job_id=job_id,
                page_number=1,
                page_size=10,
            ),
        )
        body = status_resp.body.to_map()
        print(json.dumps(body, ensure_ascii=False, indent=2))
        status = body.get("Data", {}).get("Status")
        if status in {"COMPLETED", "FAILED"}:
            return
        time.sleep(2)
    raise RuntimeError(f"Job {job_id} did not finish in time")


def retrieve(client: BailianClient, workspace_id: str, index_id: str, query: str) -> None:
    resp = client.retrieve(
        workspace_id,
        bailian_models.RetrieveRequest(
            index_id=index_id,
            query=query,
            dense_similarity_top_k=5,
            sparse_similarity_top_k=5,
            enable_reranking=False,
            save_retriever_history=False,
        ),
    )
    body = resp.body.to_map()
    print("Retrieve result:")
    print(json.dumps(body, ensure_ascii=False, indent=2))


def main() -> int:
    workspace_id = os.getenv("ALIBABA_CLOUD_WORKSPACE_ID")
    category_id = os.getenv("BAILIAN_CATEGORY_ID")
    index_id = os.getenv("BAILIAN_INDEX_ID")
    file_path = os.getenv("BAILIAN_FILE_PATH")
    query = os.getenv("BAILIAN_QUERY", "¿De qué trata el documento?")

    if not workspace_id:
        print("Missing ALIBABA_CLOUD_WORKSPACE_ID")
        return 1
    if not category_id:
        print("Missing BAILIAN_CATEGORY_ID")
        return 1
    if not index_id:
        print("Missing BAILIAN_INDEX_ID")
        return 1
    if not file_path:
        print("Missing BAILIAN_FILE_PATH")
        return 1
    file_path = file_path.strip().strip("'\"")
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return 1

    client = build_client()
    file_name = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        data = f.read()
    md5 = hashlib.md5(data).hexdigest()
    size_in_bytes = str(len(data))

    try:
        lease_resp = client.apply_file_upload_lease(
            category_id,
            workspace_id,
            bailian_models.ApplyFileUploadLeaseRequest(
                category_type="UNSTRUCTURED",
                file_name=file_name,
                md_5=md5,
                size_in_bytes=size_in_bytes,
            ),
        )
        lease_map = lease_resp.body.to_map()
        print("Upload lease:")
        print(json.dumps(lease_map, ensure_ascii=False, indent=2))

        if not response_ok(lease_resp):
            print(
                "Upload lease was rejected. This usually means the account/region does not match "
                "the workspace region, or the data-center feature is not enabled for this account."
            )
            return 2

        lease_data = lease_resp.body.data
        if not lease_data or not lease_data.param:
            print("Upload lease response did not include upload parameters.")
            return 2
        http_upload(
            lease_data.param.url,
            lease_data.param.method,
            lease_data.param.headers,
            file_path,
        )

        add_file_resp = client.add_file(
            workspace_id,
            bailian_models.AddFileRequest(
                category_id=category_id,
                category_type="UNSTRUCTURED",
                lease_id=lease_data.file_upload_lease_id,
                parser="AUTO_SELECT",
            ),
        )
        add_file_map = add_file_resp.body.to_map()
        print("Added file:")
        print(json.dumps(add_file_map, ensure_ascii=False, indent=2))

        submit_resp = client.submit_index_add_documents_job(
            workspace_id,
            bailian_models.SubmitIndexAddDocumentsJobRequest(
                index_id=index_id,
                category_ids=[category_id],
                source_type="DATA_CENTER_CATEGORY",
            ),
        )
        submit_map = submit_resp.body.to_map()
        print("Submitted index job:")
        print(json.dumps(submit_map, ensure_ascii=False, indent=2))

        job_id = submit_resp.body.data.id
        wait_for_job(client, workspace_id, index_id, job_id)
        retrieve(client, workspace_id, index_id, query)
        return 0
    except Exception as exc:
        print(f"Request failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
