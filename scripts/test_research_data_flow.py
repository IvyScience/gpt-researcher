#!/usr/bin/env python3
"""
测试 Query → Retriever → LLM 的完整数据流。

目的：排查为什么真实 pipeline 中 LLM 收到的是 exampleurl 而非真实 source。
- 跑一遍 conduct_research， dump 传给 LLM 的 context 和 research_sources
- 校验 Source/Title/Content 格式是否正确
- 对比 research_sources 与 context 中的 Source URLs

用法（在 Docker 内执行）:
  # 1. 简单 research（与 API 默认 flow 一致）
  docker exec gpt-researcher-1-gpt-researcher-1 python3 scripts/test_research_data_flow.py -q "数字平台居家养老服务护理员研究"

  # 2. 输出到 JSON 方便对比
  docker exec gpt-researcher-1-gpt-researcher-1 python3 scripts/test_research_data_flow.py \\
    -q "数字平台居家养老" --out-json outputs/data_flow_dump.json

  # 3. 指定 report_type（detailed 会跑 subtopic 研究，耗时较长）
  docker exec gpt-researcher-1-gpt-researcher-1 python3 scripts/test_research_data_flow.py \\
    -q "数字平台护理员" --report-type detailed

  # 4. 传入 user_id（internal_biblio/noteexpress 等 retriever 必需）
  docker exec gpt-researcher-1-gpt-researcher-1 python3 scripts/test_research_data_flow.py -q "数字平台护理员" --user-id 1

环境变量（与 .env 一致）:
  DEFAULT_RETRIEVERS, REPORT_SOURCE, LANGUAGE, TEST_USER_ID（可替代 --user-id）
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass


def _extract_source_urls_from_context(context: str) -> set:
    """与 markdown_processing.extract_source_urls_from_context 保持一致"""
    sources = set()
    if not context:
        return sources
    for m in re.finditer(r"(?mi)^\s*Source:\s*(?P<url>\S+)\s*$", context):
        u = m.group("url").strip()
        if u and u.lower() not in ("none", "null", "n/a"):
            sources.add(u)
    # MCP 格式: *Source: title (url)* - 简单提取括号内的 url
    for m in re.finditer(r"\*Source:\s*[^*]*\(([^)]+)\)\*", context):
        u = m.group(1).strip()
        if u and u.lower() not in ("none", "null", "n/a"):
            sources.add(u)
    return sources


def _summary(obj, max_len: int = 200) -> str:
    s = str(obj)
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


async def run_research_and_dump(
    query: str,
    report_type: str = "research_report",
    report_source: str = None,
    user_id: str | int | None = None,
    headers: dict = None,
    out_json: str = None,
) -> dict:
    from gpt_researcher import GPTResearcher
    from gpt_researcher.utils.enum import ReportSource, Tone

    report_source = report_source or os.getenv("REPORT_SOURCE", ReportSource.Web.value)
    headers = headers or {}

    if user_id is not None:
        headers["user_id"] = str(user_id)
    elif "user_id" not in headers and os.getenv("TEST_USER_ID"):
        headers["user_id"] = os.getenv("TEST_USER_ID")

    if "retrievers" not in headers and "retriever" not in headers:
        default = os.getenv(
            "DEFAULT_RETRIEVERS",
            "internal_biblio,internal_highlight,internal_file,noteexpress",
        )
        headers["retrievers"] = default

    researcher = GPTResearcher(
        query=query,
        report_type=report_type,
        report_source=report_source,
        tone=Tone.Objective,
        headers=headers,
        verbose=True,
    )

    print("=" * 60)
    print("Running conduct_research()...")
    print("=" * 60)

    await researcher.conduct_research()

    research_sources = researcher.get_research_sources()
    context = researcher.context
    visited_urls = set(researcher.visited_urls or [])

    if isinstance(context, list):
        context_str = "\n\n".join(str(c) for c in context) if context else ""
    else:
        context_str = str(context) if context else ""

    extracted_urls = _extract_source_urls_from_context(context_str)

    sources_dump = []
    for i, s in enumerate(research_sources or []):
        if not isinstance(s, dict):
            continue
        url = s.get("url") or s.get("href") or ""
        title = s.get("title") or ""
        raw = s.get("raw_content") or s.get("content") or ""
        sources_dump.append({
            "idx": i + 1,
            "url": url,
            "title": _summary(title, 80),
            "raw_content_preview": _summary(raw, 150),
            "raw_content_len": len(raw),
        })

    dump = {
        "query": query,
        "report_type": report_type,
        "report_source": report_source,
        "timestamp": datetime.now().isoformat(),
        "research_sources_count": len(sources_dump),
        "research_sources": sources_dump,
        "visited_urls": list(visited_urls),
        "visited_urls_count": len(visited_urls),
        "context_length": len(context_str),
        "context_preview": context_str[:4000] + ("..." if len(context_str) > 4000 else ""),
        "extracted_source_urls_from_context": list(extracted_urls),
        "extracted_urls_count": len(extracted_urls),
    }

    # 控制台输出
    print("\n=== RESEARCH_SOURCES (传给 context 的原始数据) ===")
    for s in sources_dump:
        print(f"  [{s['idx']}] url={s['url']}")
        print(f"       title={s['title']}")
        print(f"       raw_preview={s['raw_content_preview']}")

    print("\n=== VISITED_URLS (citation allowlist) ===")
    for u in sorted(visited_urls):
        print(f"  - {u}")

    print("\n=== CONTEXT 中提取的 Source URLs ===")
    for u in sorted(extracted_urls):
        print(f"  - {u}")

    print("\n=== CONTEXT 预览 (前 2500 字符) ===")
    print(context_str[:2500])
    if len(context_str) > 2500:
        print("... [truncated]")

    if not extracted_urls and sources_dump:
        print("\n⚠️ 警告: research_sources 有数据，但 context 中未提取到 Source URL。")
        print("   可能原因: context 格式不是 'Source: <url>\\nTitle: ...\\nContent: ...'")

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
        print(f"\n已保存到 {out_path}")

    return dump


async def run_detailed_one_subtopic(
    query: str,
    user_id: str | int | None = None,
    headers: dict = None,
    out_json: str = None,
) -> dict:
    """跑 detailed report 的第一个 subtopic，dump 其 context。"""
    from gpt_researcher import GPTResearcher
    from gpt_researcher.utils.enum import ReportType, ReportSource, Tone

    headers = headers or {}

    if user_id is not None:
        headers["user_id"] = str(user_id)
    elif "user_id" not in headers and os.getenv("TEST_USER_ID"):
        headers["user_id"] = os.getenv("TEST_USER_ID")

    if "retrievers" not in headers and "retriever" not in headers:
        default = os.getenv(
            "DEFAULT_RETRIEVERS",
            "internal_biblio,internal_highlight,internal_file,noteexpress",
        )
        headers["retrievers"] = default

    researcher = GPTResearcher(
        query=query,
        report_type=ReportType.DetailedReport.value,
        report_source=os.getenv("REPORT_SOURCE", ReportSource.Web.value),
        tone=Tone.Objective,
        headers=headers,
        verbose=True,
    )

    print("=" * 60)
    print("Running conduct_research() for detailed report (initial + subtopics)...")
    print("=" * 60)

    await researcher.conduct_research()
    initial_sources = researcher.get_research_sources()
    initial_context = researcher.context
    initial_visited = set(researcher.visited_urls or [])

    if isinstance(initial_context, list):
        context_str = "\n\n".join(str(c) for c in initial_context) if initial_context else ""
    else:
        context_str = str(initial_context) if initial_context else ""

    extracted = _extract_source_urls_from_context(context_str)

    dump = {
        "query": query,
        "report_type": "detailed_report",
        "phase": "initial_research",
        "timestamp": datetime.now().isoformat(),
        "research_sources_count": len(initial_sources or []),
        "research_sources": [
            {
                "url": s.get("url") or s.get("href"),
                "title": _summary(s.get("title", ""), 80),
                "raw_preview": _summary(s.get("raw_content", ""), 150),
            }
            for s in (initial_sources or []) if isinstance(s, dict)
        ],
        "visited_urls": list(initial_visited),
        "context_preview": context_str[:4000],
        "extracted_source_urls": list(extracted),
    }

    print("\n=== INITIAL RESEARCH (Detailed Report 第一步) ===")
    print("research_sources:", len(initial_sources or []))
    print("visited_urls:", len(initial_visited))
    print("extracted Source URLs:", len(extracted))
    print("\nContext 预览:")
    print(context_str[:2000])
    if len(context_str) > 2000:
        print("... [truncated]")

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
        print(f"\n已保存到 {out_json}")

    return dump


def main():
    parser = argparse.ArgumentParser(description="测试 query → retriever → LLM 数据流")
    parser.add_argument("-q", "--query", required=True, help="Research query")
    parser.add_argument(
        "--user-id", "-u", type=str, default=None,
        help="User ID (internal_biblio/noteexpress 等 retriever 必需)",
    )
    parser.add_argument("--report-type", default="research_report", choices=["research_report", "detailed"])
    parser.add_argument("--report-source", default=None, help="Override REPORT_SOURCE")
    parser.add_argument("--out-json", "-o", help="输出 JSON 文件路径")
    args = parser.parse_args()

    user_id = args.user_id or os.getenv("TEST_USER_ID")

    if args.report_type == "detailed":
        asyncio.run(run_detailed_one_subtopic(args.query, user_id=user_id, out_json=args.out_json))
    else:
        asyncio.run(run_research_and_dump(
            args.query,
            report_type="research_report",
            report_source=args.report_source,
            user_id=user_id,
            out_json=args.out_json,
        ))


if __name__ == "__main__":
    main()
