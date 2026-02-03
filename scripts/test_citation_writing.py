#!/usr/bin/env python3
"""
Standalone script to test citation/writing flow in isolation.

Purpose: Debug why LLM cites exampleurl/placeholder instead of real sources.
- Inspect exactly what context is sent to the LLM
- Verify Source/Title/Content format in context
- See raw LLM output before post-processing

Usage:
  # 1. 使用 JSON 文件中的 mock context（与真实 context 格式一致）
  python scripts/test_citation_writing.py --context-json fixtures/sample_context.json

  # 2. 使用内联 context 字符串
  python scripts/test_citation_writing.py --context "Source: https://doi.org/10.1234/abc
  Title: 示例论文
  Content: 这是摘要内容..."

  # 3. 从 FAISS 持久化目录加载样本（若存在）
  python scripts/test_citation_writing.py --faiss-path ./faiss_index --query "数字平台居家养老"

  # 4. 仅校验 context 格式和提取的 Source URLs（不调用 LLM）
  python scripts/test_citation_writing.py --context-json fixtures/sample_context.json --dry-run

  # 5. 从 fixtures 创建 FAISS 索引并保存，便于后续用 --faiss-path 测试
  python scripts/test_citation_writing.py --save-faiss ./faiss_sample --context-json fixtures/sample_context.json

Vector Store 持久化说明:
  - InMemoryVectorStore: 纯内存，进程退出即丢失
  - FAISS: 可持久化，save_local(path) / load_local(path, embeddings)
  - 本项目默认使用 InMemoryVectorStore，需在 research 时显式 save 才能复现
  - 使用 --save-faiss 可从 fixtures 创建样本索引，便于调试
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass


def _format_context_from_sources(sources: list[dict]) -> str:
    """Format research_sources into the same string format sent to LLM (Source/Title/Content). :-)"""
    lines = []
    for s in sources:
        url = s.get("url") or s.get("href") or s.get("id") or ""
        title = s.get("title") or ""
        content = s.get("raw_content") or s.get("content") or s.get("abstract") or ""
        lines.append(f"Source: {url}\nTitle: {title}\nContent: {content}")
    return "\n\n".join(lines)


def _extract_sources_from_context(context: str) -> set:
    """Use the same extraction logic as markdown_processing. :-)"""
    import re
    sources = set()
    if not context:
        return sources
    for m in re.finditer(r"(?mi)^\s*Source:\s*(?P<url>\S+)\s*$", context):
        u = m.group("url").strip()
        if u and u.lower() not in ("none", "null", "n/a"):
            sources.add(u)
    return sources


async def run_subtopic_report(
    query: str,
    context: str,
    main_topic: str = "",
    language: str = "chinese",
    dry_run: bool = False,
) -> str:
    from gpt_researcher.config.config import Config
    from gpt_researcher.actions.report_generation import generate_report
    from gpt_researcher.utils.enum import Tone

    cfg = Config()
    cfg.language = language or getattr(cfg, "language", "english")

    if dry_run:
        return "[DRY RUN - no LLM call]"

    report = await generate_report(
        query=query,
        context=context,
        agent_role_prompt=cfg.agent_role or "You are a professional research analyst.",
        report_type="subtopic_report",
        tone=Tone.Objective,
        report_source="web",
        websocket=None,
        cfg=cfg,
        main_topic=main_topic or query,
        existing_headers=[],
        relevant_written_contents=[],
        cost_callback=None,
        prompt_family=__import__("gpt_researcher.prompts", fromlist=["PromptFamily"]).PromptFamily,
    )
    return report or ""


async def run_introduction(
    query: str,
    context: str,
    language: str = "chinese",
    dry_run: bool = False,
) -> str:
    from gpt_researcher.config.config import Config
    from gpt_researcher.actions.report_generation import write_report_introduction

    cfg = Config()
    cfg.language = language or getattr(cfg, "language", "english")

    if dry_run:
        return "[DRY RUN - no LLM call]"

    intro = await write_report_introduction(
        query=query,
        context=context,
        agent_role_prompt=cfg.agent_role or "You are a professional research analyst.",
        config=cfg,
        websocket=None,
        cost_callback=None,
        research_gap="",
    )
    return intro or ""


def save_faiss_from_sources(sources: list[dict], out_path: str) -> None:
    """Create FAISS index from sources and save to disk for later testing. :-)"""
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=s.get("raw_content") or s.get("content") or s.get("abstract") or "",
            metadata={"source": s.get("url") or s.get("href") or "", "title": s.get("title") or ""},
        )
        for s in sources
    ]
    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(docs, embeddings)
    Path(out_path).mkdir(parents=True, exist_ok=True)
    vs.save_local(out_path)
    print(f"Saved FAISS index to {out_path} ({len(docs)} docs)", file=sys.stderr)


def load_faiss_samples(faiss_path: str, query: str, k: int = 5) -> str:
    """Load samples from FAISS index and return formatted context string. :-)"""
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    path = Path(faiss_path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS path not found: {faiss_path}")

    embeddings = OpenAIEmbeddings()
    vs = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    docs = vs.similarity_search(query, k=k)

    from gpt_researcher.prompts import PromptFamily
    return PromptFamily.pretty_print_docs(docs, top_n=k)


def main():
    parser = argparse.ArgumentParser(description="Test citation/writing with controllable context")
    parser.add_argument("--context", "-c", type=str, help="Raw context string (Source/Title/Content format)")
    parser.add_argument("--context-json", "-j", type=str, help="Path to JSON file with sources list")
    parser.add_argument(
        "--faiss-path", "-f", type=str,
        help="Path to FAISS index; will run similarity_search with --query",
    )
    parser.add_argument(
        "--query", "-q", type=str,
        default="数字平台居家养老服务护理员研究",
        help="Query / subtopic for report",
    )
    parser.add_argument("--main-topic", "-m", type=str, default="", help="Main topic for subtopic report")
    parser.add_argument("--language", "-l", type=str, default="chinese", help="Output language")
    parser.add_argument("--mode", choices=["subtopic", "introduction"], default="subtopic", help="Report mode")
    parser.add_argument("--dry-run", action="store_true", help="Only validate context format, do not call LLM")
    parser.add_argument("--out", "-o", type=str, help="Write report output to file")
    parser.add_argument(
        "--dump-context", action="store_true", default=True,
        help="Print context and extracted sources (default: True)",
    )
    parser.add_argument("--save-faiss", type=str, help="Create FAISS index from --context-json and save to path")
    args = parser.parse_args()

    if args.save_faiss:
        if not args.context_json:
            print("Error: --save-faiss requires --context-json", file=sys.stderr)
            sys.exit(1)
        p = Path(args.context_json)
        if not p.exists():
            p = PROJECT_ROOT / "fixtures" / args.context_json
        if not p.exists():
            print(f"Error: Context JSON not found: {args.context_json}", file=sys.stderr)
            sys.exit(1)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        sources = data.get("sources", data) if isinstance(data, dict) else data
        if isinstance(sources, list):
            save_faiss_from_sources(sources, args.save_faiss)
        else:
            print("Error: JSON must contain 'sources' list", file=sys.stderr)
            sys.exit(1)
        return

    context = ""
    if args.context:
        context = args.context.strip()
    elif args.context_json:
        p = Path(args.context_json)
        if not p.exists():
            # Try relative to project fixtures
            p = PROJECT_ROOT / "fixtures" / args.context_json
        if not p.exists():
            print(f"Error: Context JSON not found: {args.context_json}", file=sys.stderr)
            sys.exit(1)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            context = _format_context_from_sources(data)
        elif isinstance(data, dict):
            sources = (
                data.get("sources")
                or data.get("context")
                or data.get("research_sources")
                or []
            )
            ctx_str = data.get("context_str", "")
            context = _format_context_from_sources(sources) if sources else ctx_str
        else:
            context = str(data)
    elif args.faiss_path:
        context = load_faiss_samples(args.faiss_path, args.query, k=5)
        print(f"[Loaded {len(context)} chars from FAISS]", file=sys.stderr)
    else:
        # Default: minimal sample to verify flow
        context = (
            "Source: https://doi.org/10.13858/j.cnki.cn32-1312/c.20250718.011\n"
            "Title: 技术嵌入与制度吸纳：居家养老服务平台化转型的底层逻辑\n"
            "Content: 居家养老服务平台化转型，是以数智技术赋能养老服务效能的关键策略。"
            '基于对江苏多地智慧居家养老服务平台的调研，构建"技术嵌入-制度吸纳"分析框架。'
            "研究发现，居家养老服务平台化转型呈现数据赋能流于理念、考核僵化服务内卷等特征。"
        )

    if not context:
        print("Error: No context provided. Use --context, --context-json, or --faiss-path.", file=sys.stderr)
        sys.exit(1)

    extracted = _extract_sources_from_context(context)
    if args.dump_context:
        print("=== CONTEXT (first 2000 chars) ===")
        print(context[:2000])
        if len(context) > 2000:
            print("... [truncated]")
        print("\n=== EXTRACTED SOURCE URLS ===")
        for u in sorted(extracted):
            print(f"  - {u}")
        if not extracted:
            print("  (none - check 'Source: <url>' format in context)")
        print()

    if args.dry_run:
        print("[DRY RUN] Context format validated. Exiting without LLM call.")
        return

    async def run():
        if args.mode == "subtopic":
            return await run_subtopic_report(
                query=args.query,
                context=context,
                main_topic=args.main_topic or args.query,
                language=args.language,
                dry_run=args.dry_run,
            )
        return await run_introduction(
            query=args.query,
            context=context,
            language=args.language,
            dry_run=args.dry_run,
        )

    report = asyncio.run(run())

    print("=== LLM OUTPUT (raw, no citation post-processing) ===")
    print(report)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nWritten to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
