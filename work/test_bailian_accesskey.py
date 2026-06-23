#!/usr/bin/env python3
import inspect
import json
import os
import sys

from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_bailian20231229.client import Client as BailianClient
from alibabacloud_bailian20231229 import models as bailian_models


def main() -> int:
    access_key_id = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")
    access_key_secret = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
    endpoint = os.getenv("ALIBABA_CLOUD_ENDPOINT", "bailian.cn-beijing.aliyuncs.com")
    workspace_id = os.getenv("ALIBABA_CLOUD_WORKSPACE_ID")

    if not access_key_id or not access_key_secret:
        print("Missing ALIBABA_CLOUD_ACCESS_KEY_ID or ALIBABA_CLOUD_ACCESS_KEY_SECRET")
        return 1
    if not workspace_id:
        print("Missing ALIBABA_CLOUD_WORKSPACE_ID")
        return 1

    config = open_api_models.Config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        endpoint=endpoint,
    )

    client = BailianClient(config)
    request = bailian_models.ListIndicesRequest()
    headers = {}
    runtime = util_models.RuntimeOptions()

    try:
        response = client.list_indices_with_options(workspace_id, request, headers, runtime)
        body = response.body
        if hasattr(body, "to_map"):
            print(json.dumps(body.to_map(), ensure_ascii=False, indent=2))
        else:
            print(body)
        return 0
    except TypeError as exc:
        print(f"Request failed: {exc}")
        print(f"Method signature: {inspect.signature(client.list_indices_with_options)}")
        return 2
    except Exception as exc:
        print(f"Request failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
