#!/usr/bin/env python3
"""
Fill [citation required] placeholders in a markdown file using internal_biblio and noteexpress.

Usage:
  python scripts/fill_citations.py input.md -o output.md --user-id 1
  python scripts/fill_citations.py report.md -o report_filled.md -u 1
  python scripts/fill_citations.py report.md  # writes to report_filled.md by default

Environment:
  TEST_USER_ID: default user_id (required for internal_biblio)
  INTERNAL_API_KEY, INTERNAL_API_BASE_URL, NOTEEXPRESS_API_KEY, etc.
  EMBEDDING: for rerank (default openai:text-embedding-3-small)
  CITATION_RERANK: true|false (default true)
  CITATION_LLM_VERIFY: true|false (default true) - LLM checks relevance before citing
  CITATION_SIMILARITY_THRESHOLD: min similarity 0-1 (default 0.35)
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Configure logging before any imports that use it :-)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("fill_citations")

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


async def main():
    parser = argparse.ArgumentParser(description="Fill [citation required] with retriever-backed sources")
    parser.add_argument("input", type=str, help="Input markdown file path")
    parser.add_argument("-o", "--output", type=str, help="Output markdown file (default: input_filled.md)")
    parser.add_argument("-u", "--user-id", type=str, default=None, help="User ID (required for internal_biblio)")
    parser.add_argument("--max-sources", type=int, default=1, help="Max sources per placeholder (default: 1, 最佳)")
    parser.add_argument("--style", choices=["paren", "inline"], default="inline",
                        help="Citation format: inline=[Title](url) (default), paren=([Title](url))")
    parser.add_argument("--no-replace-existing", action="store_true",
                        help="Do NOT replace existing [Source](url) links with [citation required] first")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-rerank", action="store_true", help="Disable embedding rerank (use retriever order)")
    parser.add_argument("--no-llm-verify", action="store_true", help="Disable LLM relevance verification")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting fill_citations...")
    user_id = args.user_id or os.getenv("TEST_USER_ID")
    if not user_id:
        print("Error: --user-id or TEST_USER_ID required for internal_biblio", file=sys.stderr)
        sys.exit(1)

    inp = Path(args.input)
    if not inp.exists():
        print(f"Error: Input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    md_text = inp.read_text(encoding="utf-8")
    logger.info("Read %d chars from %s", len(md_text), inp)

    logger.info("Loading citation_filling module...")
    from gpt_researcher.actions.citation_filling import fill_citations

    headers = {}
    if os.getenv("INTERNAL_API_KEY"):
        headers["internal_api_key"] = os.getenv("INTERNAL_API_KEY")
    if os.getenv("INTERNAL_API_BASE_URL"):
        headers["internal_api_base_url"] = os.getenv("INTERNAL_API_BASE_URL")
    if os.getenv("NOTEEXPRESS_API_KEY"):
        headers["noteexpress_api_key"] = os.getenv("NOTEEXPRESS_API_KEY")
    if os.getenv("NOTEEXPRESS_BASE_URL"):
        headers["noteexpress_base_url"] = os.getenv("NOTEEXPRESS_BASE_URL")

    logger.info("Filling citations (user_id=%s, replace_existing=%s)...", user_id, not args.no_replace_existing)
    os.environ["CITATION_RERANK"] = "false" if args.no_rerank else "true"
    os.environ["CITATION_LLM_VERIFY"] = "false" if args.no_llm_verify else "true"
    filled = await fill_citations(
        md_text,
        user_id=user_id,
        headers=headers,
        max_sources_per_placeholder=args.max_sources,
        citation_style=args.style,
        replace_existing_citations=not args.no_replace_existing,
        verbose=args.verbose,
    )

    out_path = Path(args.output or inp.parent / f"{inp.stem}_filled{inp.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(filled, encoding="utf-8")
    logger.info("Done. Written to %s", out_path)


if __name__ == "__main__":
    asyncio.run(main())
