#!/usr/bin/env python3
import json
import os
import sys

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


def main() -> int:
    workspace_id = os.getenv("ALIBABA_CLOUD_WORKSPACE_ID")
    if not workspace_id:
        print("Missing ALIBABA_CLOUD_WORKSPACE_ID")
        return 1

    client = build_client()

    category_name = os.getenv("BAILIAN_CATEGORY_NAME", "rag-demo-category")
    index_name = os.getenv("BAILIAN_INDEX_NAME", "rag-demo-index")

    try:
        add_category_req = bailian_models.AddCategoryRequest(
            category_name=category_name,
            category_type="UNSTRUCTURED",
        )
        add_category_resp = client.add_category(workspace_id, add_category_req)
        category_id = add_category_resp.body.data.category_id
        print(f"Created category: {category_id}")
        dump(add_category_resp)

        create_index_req = bailian_models.CreateIndexRequest(
            name=index_name,
            sink_type="BUILT_IN",
            source_type="DATA_CENTER_CATEGORY",
            structure_type="unstructured",
            category_ids=[category_id],
        )
        create_index_resp = client.create_index(workspace_id, create_index_req)
        print("Created knowledge base:")
        dump(create_index_resp)
        return 0
    except Exception as exc:
        print(f"Request failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
