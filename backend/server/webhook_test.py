import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, Optional

import httpx


def _build_payload(
    research_id: str,
    status: str,
    report: str,
    error: Optional[str],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "research_id": research_id,
        "status": status,
        "timestamp": int(time.time()),
    }

    if status == "failed":
        if error:
            payload["error"] = error
        payload["data"] = {
            "report": "",
            "md_path": "",
            "docx_path": "",
            "pdf_path": "",
            "research_information": {
                "research_costs": 0.0,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "source_urls": [],
                "visited_urls": [],
                "research_images": [],
            },
        }
    else:
        payload["data"] = {
            "report": report,
            "md_path": "",
            "docx_path": "",
            "pdf_path": "",
            "research_information": {
                "research_costs": 0.0,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "source_urls": [],
                "visited_urls": [],
                "research_images": [],
            },
        }

    if extra:
        for k, v in extra.items():
            payload[k] = v

    return payload


async def _post(
    url: str,
    payload: Dict[str, Any],
    api_key: Optional[str],
    host_header: Optional[str],
    trust_env: bool,
) -> httpx.Response:
    headers: Dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key
    if host_header:
        headers["Host"] = host_header

    async with httpx.AsyncClient(timeout=10.0, trust_env=trust_env) as client:
        return await client.post(url, json=payload, headers=headers)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send a test webhook request (same schema as backend/server/app.py)."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("WEBHOOK_URL", ""),
        help="Webhook URL (or set WEBHOOK_URL).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("WEBHOOK_API_KEY", ""),
        help="x-api-key header (or set WEBHOOK_API_KEY).",
    )
    parser.add_argument(
        "--host-header",
        type=str,
        default=os.getenv("WEBHOOK_HOST_HEADER", ""),
        help="Override Host header (or set WEBHOOK_HOST_HEADER).",
    )
    parser.add_argument(
        "--research-id",
        type=str,
        default="webhook_test",
        help="research_id to send.",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="completed",
        choices=["completed", "failed"],
        help="status to send.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="webhook test report",
        help="report content (for completed).",
    )
    parser.add_argument(
        "--error",
        type=str,
        default="webhook test error",
        help="error content (for failed).",
    )
    parser.add_argument(
        "--extra-json",
        type=str,
        default="",
        help="Extra top-level JSON to merge into payload.",
    )
    parser.add_argument(
        "--trust-env",
        action="store_true",
        help="Allow httpx to use env proxies (default: disabled).",
    )
    args = parser.parse_args()

    if not args.url:
        raise SystemExit("Missing webhook url: pass --url or set WEBHOOK_URL")

    extra: Optional[Dict[str, Any]] = None
    if args.extra_json:
        extra = json.loads(args.extra_json)
        if not isinstance(extra, dict):
            raise SystemExit("--extra-json must be a JSON object")

    payload = _build_payload(
        research_id=args.research_id,
        status=args.status,
        report=args.report,
        error=args.error if args.status == "failed" else None,
        extra=extra,
    )

    resp = await _post(
        url=args.url,
        payload=payload,
        api_key=args.api_key or None,
        host_header=args.host_header or None,
        trust_env=bool(args.trust_env),
    )

    body = resp.text
    preview = body[:2000]

    print(f"URL: {args.url}")
    print(f"Status: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('content-type', '')}")
    print("Response (first 2000 chars):")
    print(preview)

    if resp.status_code >= 400:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
