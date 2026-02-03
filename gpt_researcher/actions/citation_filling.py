"""
Citation filling: replace [citation required] with retriever-backed sources.

Flow: extract claim -> retrievers -> embedding rerank -> LLM relevance check -> fill.
LLM verifies which of top 2 sources (if any) actually supports the claim; if none, leaves [citation required]. :-)
"""

import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

PLACEHOLDER = "[citation required]"
PLACEHOLDER_RE = re.compile(re.escape(PLACEHOLDER), re.IGNORECASE)
# Markdown link: [text](url), used to replace existing citations with placeholder
MD_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]+\)")


def _is_retracted(source: dict) -> bool:
    """True if source title contains 'retracted' (case-insensitive). :-)"""
    title = (source.get("title") or "").strip()
    return "retracted" in title.lower()


def _replace_existing_citations_with_placeholder(md_text: str) -> str:
    """
    Replace all markdown citation links [text](url) with [citation required].
    Then collapse redundant wrappers like ([citation required]), **([citation required])**, etc. :-)
    """
    text = MD_LINK_RE.sub(PLACEHOLDER, md_text)
    # Repeat cleanup until stable (handles nested cases like **([A]; [B])**)
    for _ in range(5):
        prev = text
        text = re.sub(r"\[citation required\]\s*[;，,]\s*\[citation required\]", PLACEHOLDER, text, flags=re.IGNORECASE)
        text = re.sub(r"\(\s*\[citation required\]\s*\)", PLACEHOLDER, text, flags=re.IGNORECASE)
        text = re.sub(r"\*\*\s*\[citation required\]\s*\*\*", PLACEHOLDER, text, flags=re.IGNORECASE)
        if text == prev:
            break
    return text


def _get_embeddings():
    """Lazy init embeddings from Config or EMBEDDING env (provider:model). :-)"""
    embed = getattr(_get_embeddings, "_cached", None)
    if embed is not None:
        return embed
    try:
        from gpt_researcher.config import Config
        cfg = Config()
        from gpt_researcher.memory.embeddings import Memory
        mem = Memory(cfg.embedding_provider, cfg.embedding_model, **cfg.embedding_kwargs)
        embed = mem.get_embeddings()
    except Exception as e:
        logger.warning("Config/Memory init failed, using OpenAI default: %s", e)
        embedding_str = os.getenv("EMBEDDING", "openai:text-embedding-3-small")
        parts = embedding_str.split(":", 1)
        provider = parts[0] if parts else "openai"
        model = parts[1] if len(parts) > 1 else "text-embedding-3-small"
        from gpt_researcher.memory.embeddings import Memory
        mem = Memory(provider, model)
        embed = mem.get_embeddings()
    _get_embeddings._cached = embed
    return embed


def _rerank_by_similarity(claim_text: str, results: List[dict], top_k: int, embeddings) -> List[dict]:
    """
    Rerank results by embedding similarity between claim and each result's content. :-)
    Uses cosine similarity; returns top_k results. Filters out results below threshold.
    """
    if not results or not claim_text:
        return results[:top_k]
    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy not available, skipping rerank")
        return results[:top_k]
    threshold = float(os.getenv("CITATION_SIMILARITY_THRESHOLD", "0.35"))
    texts = []
    for r in results:
        title = r.get("title") or ""
        body = r.get("raw_content") or r.get("body") or ""
        texts.append(f"{title}\n{body}"[:2000])
    try:
        claim_vec = embeddings.embed_query(claim_text)
        doc_vecs = embeddings.embed_documents(texts)
    except Exception as e:
        logger.warning("Embedding failed, skipping rerank: %s", e)
        return results[:top_k]
    claim_arr = np.array(claim_vec, dtype=float)
    scores = []
    for i, dv in enumerate(doc_vecs):
        doc_arr = np.array(dv, dtype=float)
        norm = np.linalg.norm(claim_arr) * np.linalg.norm(doc_arr)
        sim = float(np.dot(claim_arr, doc_arr) / norm) if norm > 1e-9 else 0.0
        scores.append((i, sim))
    scores.sort(key=lambda x: -x[1])
    reranked = [results[s[0]] for s in scores[:top_k] if s[1] >= threshold]
    # Best-only: if best score is not clearly above rest, return at most 1 to avoid bad citations
    best_only = os.getenv("CITATION_BEST_ONLY", "true").strip().lower() in ("1", "true", "yes", "y", "on")
    if best_only and len(reranked) > 1 and len(scores) >= 2:
        best_sc, second_sc = scores[0][1], scores[1][1]
        if best_sc - second_sc < 0.08:
            reranked = reranked[:1]
    if not reranked and scores:
        best = scores[0][1]
        logger.debug("All results below threshold %.2f (best=%.4f), leaving placeholder", threshold, best)
    if logger.isEnabledFor(logging.DEBUG):
        for j, (idx, sc) in enumerate(scores[:min(5, len(scores))]):
            u = results[idx].get("url", "")[:50]
            logger.debug("Rerank %d: score=%.4f (threshold=%.2f) url=%s", j + 1, sc, threshold, u)
    return reranked


def _extract_claim_text(md_text: str, placeholder_pos: int, max_chars: int = 300) -> str:
    """
    Extract the claim text for retrieval - text around the placeholder. :-)
    """
    start = max(0, placeholder_pos - 350)
    end = min(len(md_text), placeholder_pos + 350)
    window = md_text[start:end]
    window = re.sub(PLACEHOLDER_RE, " ", window)
    text = re.sub(r"\s+", " ", window)
    text = re.sub(r"[#*_\[\]`]", "", text).strip()
    return text[:max_chars] if text else "research"


async def _search_with_retrievers(
    claim_text: str,
    user_id: str,
    headers: dict,
    max_results: int = 4,
    use_rerank: bool = True,
) -> List[dict]:
    """
    Call internal_biblio and noteexpress, merge, dedupe, then rerank by embedding similarity. :-)
    """
    import asyncio

    pool_size = 12 if use_rerank else max_results
    results = []
    headers = dict(headers or {})
    headers["user_id"] = str(user_id)

    retriever_names = ["internal_biblio", "noteexpress"]
    for name in retriever_names:
        try:
            from gpt_researcher.actions.retriever import get_retriever

            Klass = get_retriever(name)
            if Klass is None:
                logger.debug("Retriever %s not available, skipping", name)
                continue
            retriever = Klass(query=claim_text, headers=headers)
            logger.debug("Querying %s (query=%.60s...)", name, claim_text)
            items = await asyncio.to_thread(retriever.search, max_results=pool_size)
            for item in (items or []):
                if not isinstance(item, dict):
                    continue
                url = item.get("url") or item.get("href")
                title = item.get("title") or item.get("raw_content", "")[:80] or "Source"
                cand = {
                    "url": str(url),
                    "title": str(title).strip() or "Source",
                    "raw_content": item.get("raw_content", ""),
                    "body": item.get("body", ""),
                    "author": item.get("author", ""),
                    "year": item.get("year", ""),
                }
                if _is_retracted(cand):
                    logger.debug("Excluding retracted paper: %s", (title or url)[:60])
                    continue
                if url and url not in {r.get("url") for r in results}:
                    results.append(cand)
        except Exception as e:
            logger.warning("Citation retriever %s failed: %s", name, e)

    if use_rerank and len(results) >= 1:
        try:
            embeddings = _get_embeddings()
            results = _rerank_by_similarity(claim_text, results, top_k=max_results, embeddings=embeddings)
            logger.debug("Reranked to top %d", len(results))
        except Exception as e:
            logger.warning("Rerank failed, using retriever order: %s", e)
            results = results[:max_results]
    else:
        results = results[:max_results]

    return results


def _format_author_year(source: dict) -> str:
    """Format as 'Author, Year' or 'Author et al., Year' for citation label. :-)"""
    author = (source.get("author") or "").strip()
    year = (source.get("year") or "").strip()
    if author and year:
        return f"{author}, {year}"
    if author:
        return f"{author}, n.d." if not year else author
    if year:
        return f"Source, {year}"
    return "Source"


async def _llm_verify_relevance(
    claim_text: str,
    sources: List[dict],
    cfg=None,
    cost_callback=None,
) -> Optional[dict]:
    """
    Ask LLM to verify which source (if any) actually supports the claim.
    Returns 文献1 (sources[0]), 文献2 (sources[1]), or None. :-)
    """
    if not sources or not claim_text:
        return None
    if len(sources) == 1:
        raw = (sources[0].get("raw_content") or sources[0].get("body") or "")[:800]
        s1_content = f"Title: {sources[0].get('title', '')}\nContent: {raw}"
        prompt = f"""判断候选文献是否在语义上支持原文论述。只回答 s1 或 None。

原文论述：
{claim_text}

s1: {s1_content}"""
    else:
        r0 = (sources[0].get("raw_content") or sources[0].get("body") or "")[:600]
        r1 = (sources[1].get("raw_content") or sources[1].get("body") or "")[:600]
        s1_content = f"Title: {sources[0].get('title', '')}\nContent: {r0}"
        s2_content = f"Title: {sources[1].get('title', '')}\nContent: {r1}"
        prompt = f"""判断哪个文献在语义上支持原文论述。只回答 s1、s2 或 None。

原文论述：
{claim_text}

s1: {s1_content}

s2: {s2_content}"""

    try:
        from gpt_researcher.config import Config
        from gpt_researcher.utils.llm import create_chat_completion

        config = cfg or Config()
        smart = os.getenv("SMART_LLM", "openai:gpt-4o-mini")
        model = getattr(config, "smart_llm_model", None) or smart.split(":")[-1]
        provider = getattr(config, "smart_llm_provider", None) or smart.split(":")[0]
        llm_kw = getattr(config, "llm_kwargs", None) or {}
        resp = (
            await create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,
                max_tokens=20,
                llm_provider=provider,
                llm_kwargs=llm_kw,
                cost_callback=cost_callback,
            )
            or ""
        ).strip().lower()
        if "s2" in resp and len(sources) > 1:
            return sources[1]
        if "s1" in resp:
            return sources[0]
        return None
    except Exception as e:
        logger.warning("LLM relevance verification failed: %s", e)
        return None


def _format_citation(source: dict, style: str = "paren", use_author_year: Optional[bool] = None) -> str:
    """Format as [Author, Year](url) when possible, else [Title](url). :-)"""
    url = source.get("url", "")
    if not url:
        return ""
    if use_author_year is None:
        v = os.getenv("CITATION_USE_AUTHOR_YEAR", "true").strip().lower()
        use_author_year = v in ("1", "true", "yes", "y", "on")
    if use_author_year:
        label = _format_author_year(source)
    else:
        label = source.get("title", "Source") or "Source"
    label = re.sub(r"[\[\]\*_`]", "", str(label))[:80]
    link = f"[{label}]({url})"
    if style == "paren":
        return f"({link})"
    return link


async def fill_citations(
    md_text: str,
    user_id: str,
    headers: dict = None,
    max_sources_per_placeholder: int = 1,
    citation_style: str = "inline",
    replace_existing_citations: bool = True,
    verbose: bool = False,
    use_llm_verify: Optional[bool] = None,
    cfg=None,
    cost_callback=None,
) -> str:
    """
    Find all [citation required] in md_text, retrieve best sources, replace with citations.
    Output format: [Title](url) only (no outer parens). :-)

    Args:
        md_text: Markdown content with [citation required] placeholders or [Source](url) links.
        user_id: User ID for internal_biblio.
        headers: Optional headers (api keys, base URLs, etc).
        max_sources_per_placeholder: Max 1-2 sources per placeholder.
        citation_style: "inline" for [Title](url) (default), "paren" for ([Title](url)).
        replace_existing_citations: If True, replace [Source](url) etc. with [citation required] first.

    Returns:
        Markdown with placeholders replaced by [Title](url) citations.
    """
    if replace_existing_citations:
        before_count = len(MD_LINK_RE.findall(md_text))
        md_text = _replace_existing_citations_with_placeholder(md_text)
        if verbose or before_count:
            logger.info("Replaced %d existing citation link(s) with [citation required]", before_count)
    if PLACEHOLDER not in md_text and PLACEHOLDER.lower() not in md_text.lower():
        logger.info("No [citation required] found; output unchanged")
        return md_text

    placeholders = list(PLACEHOLDER_RE.finditer(md_text))
    logger.info("Processing %d [citation required] placeholder(s)...", len(placeholders))
    out = []
    last_end = 0
    filled_count = 0
    for i, m in enumerate(placeholders):
        out.append(md_text[last_end : m.start()])
        claim = _extract_claim_text(md_text, m.start())
        if verbose:
            logger.debug("Placeholder %d/%d query: %.80s...", i + 1, len(placeholders), claim)
        use_rerank = os.getenv("CITATION_RERANK", "true").strip().lower() in ("1", "true", "yes", "y", "on")
        llv = os.getenv("CITATION_LLM_VERIFY", "true").strip().lower()
        do_llm_verify = use_llm_verify if use_llm_verify is not None else llv in ("1", "true", "yes", "y", "on")
        fetch_k = 2 if do_llm_verify else max_sources_per_placeholder
        sources = await _search_with_retrievers(
            claim, user_id, headers,
            max_results=fetch_k,
            use_rerank=use_rerank,
        )
        if do_llm_verify and sources:
            chosen = await _llm_verify_relevance(claim, sources, cfg=cfg, cost_callback=cost_callback)
            sources = [chosen] if chosen else []
        if verbose:
            logger.info("Placeholder %d/%d: found %d source(s)", i + 1, len(placeholders), len(sources))
        if sources:
            cites = " ".join(_format_citation(s, citation_style) for s in sources)
            out.append(f" {cites} ")
            filled_count += 1
        else:
            out.append(" ")
        last_end = m.end()
    out.append(md_text[last_end:])
    logger.info("Filled %d/%d placeholder(s)", filled_count, len(placeholders))
    return "".join(out)
