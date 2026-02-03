import json
import os
import logging

from ..utils.llm import construct_subtopics
from ..utils import llm as llm_utils
from ..utils.costs import estimate_token_usage
from ..actions import (
    stream_output,
    generate_report,
    generate_draft_section_titles,
    write_report_introduction,
    write_conclusion
)
from ..actions.markdown_processing import (
    sanitize_citation_links,
    canonicalize_intext_citations,
    prune_unsupported_citation_claims,
    extract_source_urls_from_context,
)
from ..prompts import _citation_use_placeholder


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports based on research data."""

    def _looks_like_language(self, text: str, language: str) -> bool:
        """Heuristic check to avoid mixed-language outputs. :-)"""
        try:
            t = text or ""
            lang = (language or "").strip().lower()
            if not t or not lang:
                return True

            cjk = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
            latin = sum(1 for ch in t if ("a" <= ch.lower() <= "z"))

            is_zh = ("chinese" in lang) or lang.startswith("zh") or ("中文" in language)
            if is_zh:
                # Only flag as wrong-language when it's clearly mostly Latin.
                # Keep this lenient to avoid rewriting short texts (unit tests, short headings). :-)
                if cjk >= 8 and cjk >= latin:
                    return True
                if latin >= 40 and cjk == 0:
                    return False
                return True

            # Default: treat as English-like.
            if "english" in lang:
                if latin >= 8 and latin >= (cjk * 2):
                    return True
                if cjk >= 40 and latin == 0:
                    return False
                return True

            return True
        except Exception:
            return True

    async def _rewrite_markdown_in_language(self, markdown_text: str, language: str) -> str:
        """
        Rewrite/translate markdown into the target language, keeping links intact. :-)
        """
        cfg = self.researcher.cfg
        text = markdown_text or ""
        if not text:
            return text

        prompt = f"""
You are an academic editor.

Task: Rewrite the following text so that it is written ONLY in {language}.

Hard constraints:
- Preserve ALL markdown links EXACTLY as-is (do NOT change link labels or URLs).
- Do NOT add any new URLs or citations.
- Keep the markdown structure (headers, paragraphs) intact.
- Return ONLY the rewritten markdown text.

Text to rewrite:
{text}
"""
        try:
            rewritten = await llm_utils.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You rewrite text without changing links."},
                    {"role": "user", "content": prompt},
                ],
                model=cfg.logical_llm_model,
                llm_provider=cfg.logical_llm_provider,
                llm_kwargs=getattr(cfg, "llm_kwargs", None),
                max_tokens=min(int(getattr(cfg, "smart_token_limit", 4000) or 4000), 2000),
                temperature=0.2,
                stream=False,
                websocket=None,
                cost_callback=self.researcher.add_costs,
            )
            return rewritten or text
        except Exception:
            return text

    async def _ensure_output_language(self, markdown_text: str, language: str) -> str:
        """Ensure output is in target language; rewrite if needed. :-)"""
        if self._looks_like_language(markdown_text or "", language):
            return markdown_text
        rewritten = await self._rewrite_markdown_in_language(markdown_text or "", language)
        return rewritten or markdown_text

    def _coerce_context_to_text(self, ctx) -> str:
        """Normalize context into a single string for prompting/post-processing. :-)"""
        if ctx is None:
            return ""
        if isinstance(ctx, str):
            return ctx
        if isinstance(ctx, list):
            # Common cases: list[str] from asyncio.gather / vectorstore path
            try:
                return "\n\n".join([c for c in ctx if isinstance(c, str)])
            except Exception:
                return "\n\n".join([str(c) for c in ctx])
        return str(ctx)

    async def _maybe_fill_citations(self, text: str) -> str:
        """Fill [citation required] via retrievers when in placeholder mode. :-)"""
        if not _citation_use_placeholder() or "[citation required]" not in (text or ""):
            return text or ""
        user_id = (self.researcher.headers or {}).get("user_id") or os.getenv("TEST_USER_ID")
        if not user_id:
            return text or ""
        try:
            from ..actions.citation_filling import fill_citations
            return await fill_citations(
                text or "",
                user_id=str(user_id),
                headers=self.researcher.headers,
                max_sources_per_placeholder=int(os.getenv("CITATION_FILL_MAX_SOURCES", "1")),
                citation_style="inline",
                replace_existing_citations=True,
                cfg=self.researcher.cfg,
                cost_callback=self.researcher.add_costs,
            )
        except Exception as e:
            logger.warning(f"Citation filling failed: {e}")
            return text or ""

    def _allowed_urls_for_postprocessing(self, context_text: str) -> set[str] | None:
        """
        Build an allow-set for citations based on visited_urls plus any sources explicitly present
        in the context. This is used ONLY for post-processing, not for prompting. :-)
        """
        allowed_urls: set[str] = set()
        try:
            allowed_urls = set(getattr(self.researcher, "visited_urls", set()) or set())
        except Exception:
            allowed_urls = set()

        try:
            context_sources = extract_source_urls_from_context(context_text)
            if context_sources:
                allowed_urls.update(context_sources)
        except Exception:
            pass

        return allowed_urls or None

    def _append_context_source_allowlist(self, context_text: str, limit: int = 80) -> str:
        """
        Add an explicit allowlist of source URLs that already appear in the context.
        This helps the model copy/paste real URLs instead of fabricating "exampleurl1.com". :-)
        """
        try:
            sources = list(extract_source_urls_from_context(context_text) or [])
            sources = [s for s in sources if isinstance(s, str) and s.strip()]
            if not sources:
                return context_text
            sources = sources[: max(0, int(limit))]
            block = (
                "\n\nALLOWED_SOURCE_URLS (copy/paste EXACTLY one of these when citing; do NOT invent URLs):\n"
                + "\n".join(f"- {s}" for s in sources)
            )
            return f"{context_text}{block}"
        except Exception:
            return context_text

    def _extract_markdown_link_urls(self, text: str) -> set[str]:
        """Extract all markdown link URLs from a text (best-effort). :-)"""
        try:
            import re
            urls: set[str] = set()
            for m in re.finditer(r"(?<!!)\[[^\]]+\]\((?P<url>[^)]+)\)", text or ""):
                u = (m.group("url") or "").strip().strip('"').strip("'")
                if u:
                    urls.add(u)
            return urls
        except Exception:
            return set()

    async def _log_citation_postprocess_debug(
        self,
        stage: str,
        before_text: str,
        after_text: str,
        allowed_urls: set[str] | None,
    ) -> None:
        """
        Emit high-signal debug info to websocket logs so we can prove whether links were stripped by code. :-)
        """
        try:
            if not self.researcher.verbose:
                return
            before_urls = self._extract_markdown_link_urls(before_text)
            after_urls = self._extract_markdown_link_urls(after_text)
            removed = list(before_urls - after_urls)
            kept = list(after_urls)
            bare_source_before = (before_text or "").count("(Source)") + (before_text or "").count("（Source）")
            bare_source_after = (after_text or "").count("(Source)") + (after_text or "").count("（Source）")
            removed_preview = removed[:20]
            kept_preview = kept[:20]
            payload = {
                "stage": stage,
                "cfg_language": getattr(self.researcher.cfg, "language", ""),
                "citation_postprocess_enabled": _env_bool("CITATION_POSTPROCESS", True),
                "allowed_urls_count": len(allowed_urls or set()),
                "before_links": len(before_urls),
                "after_links": len(after_urls),
                "removed_links": len(removed),
                "bare_source_before": bare_source_before,
                "bare_source_after": bare_source_after,
                "removed_links_preview": removed_preview,
                "kept_links_preview": kept_preview,
            }
            await stream_output(
                "logs",
                "citation_postprocess_debug",
                json.dumps(payload, ensure_ascii=False),
                self.researcher.websocket,
                True,
                payload,
            )
        except Exception:
            return

    def __init__(self, researcher):
        self.researcher = researcher
        self.research_params = {
            "query": self.researcher.query,
            "agent_role_prompt": self.researcher.cfg.agent_role or self.researcher.role,
            "report_type": self.researcher.report_type,
            "report_source": self.researcher.report_source,
            "tone": self.researcher.tone,
            "websocket": self.researcher.websocket,
            "cfg": self.researcher.cfg,
            "headers": self.researcher.headers,
        }

    async def write_report(
        self,
        existing_headers: list = [],
        relevant_written_contents: list = [],
        ext_context=None,
        custom_prompt="",
    ) -> str:
        """
        Write a report based on existing headers and relevant contents.

        Args:
            existing_headers (list): List of existing headers.
            relevant_written_contents (list): List of relevant written contents.
            ext_context (Optional): External context, if any.
            custom_prompt (str): Custom prompt for the report.

        Returns:
            str: The generated report.
        """
        # send the selected images prior to writing report
        research_images = self.researcher.get_research_images()
        if research_images:
            await stream_output(
                "images",
                "selected_images",
                json.dumps(research_images),
                self.researcher.websocket,
                True,
                research_images
            )

        context_text = self._coerce_context_to_text(ext_context or self.researcher.context)
        # Always add context-derived allowlist to reduce fabricated URLs in citations. :-)
        context_text = self._append_context_source_allowlist(
            context_text,
            limit=int(os.getenv("CONTEXT_SOURCES_ALLOWLIST_LIMIT", "120")),
        )

        # Do NOT pass visited_urls/allowlist to the LLM by default; we enforce citations post-hoc instead :-)
        allowed_urls_for_prompt = set()
        try:
            allowed_urls_for_prompt = set(getattr(self.researcher, "visited_urls", set()) or set())
        except Exception:
            allowed_urls_for_prompt = set()

        if allowed_urls_for_prompt and _env_bool("INCLUDE_ALLOWED_SOURCES_IN_CONTEXT", False):
            limit = int(os.getenv("ALLOWED_SOURCES_LIMIT", "200"))
            allowed_list = list(allowed_urls_for_prompt)[: max(0, limit)]
            allowed_block = (
                "\n\nALLOWED_SOURCES (cite ONLY these URLs; do NOT invent papers/authors/domains; "
                "if you cannot support a claim with these sources, omit the claim):\n"
                + "\n".join(f"- {u}" for u in allowed_list)
            )
            context_text = f"{context_text}{allowed_block}"

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_report",
                f"✍️ Writing report for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        report_params = self.research_params.copy()
        report_params["context"] = context_text
        report_params["custom_prompt"] = custom_prompt

        if self.researcher.report_type == "subtopic_report":
            report_params.update({
                "main_topic": self.researcher.parent_query,
                "existing_headers": existing_headers,
                "relevant_written_contents": relevant_written_contents,
                "cost_callback": self.researcher.add_costs,
            })
        else:
            report_params["cost_callback"] = self.researcher.add_costs

        report = await generate_report(**report_params, **self.researcher.kwargs)
        raw_report = report or ""
        allowed_for_citations = self._allowed_urls_for_postprocessing(context_text)

        # Post-processing can be disabled for debugging (to prove whether links were removed by code). :-)
        if _env_bool("CITATION_POSTPROCESS", True):
            report = sanitize_citation_links(report, allowed_urls=allowed_for_citations)
            report = canonicalize_intext_citations(report, allowed_urls=allowed_for_citations)
            # Optional stricter guard: drop uncited risky claims
            if _env_bool("STRICT_CITATIONS", True):
                report = prune_unsupported_citation_claims(report)

        await self._log_citation_postprocess_debug(
            stage="report",
            before_text=raw_report,
            after_text=report or "",
            allowed_urls=allowed_for_citations,
        )

        report = await self._maybe_fill_citations(report or "")

        # Token/cost snapshot right after markdown is produced (helps debugging accounting gaps)
        try:
            log_subtopic = _env_bool("LOG_TOKEN_USAGE_SUBTOPIC", False)
            should_log = _env_bool("LOG_TOKEN_USAGE_SNAPSHOT", True) and (
                log_subtopic or self.researcher.report_type != "subtopic_report"
            )
            if should_log:
                total_usage = self.researcher.get_token_usage()
                total_cost = self.researcher.get_costs()
                # Estimate the report's own size in tokens (rough, but good sanity check)
                report_est = estimate_token_usage("", report or "", model=self.researcher.cfg.smart_llm_model)
                snapshot = {
                    "phase": "report_generated",
                    "report_type": self.researcher.report_type,
                    "model": self.researcher.cfg.smart_llm_model,
                    "report_chars": len(report or ""),
                    "report_bytes": len((report or "").encode("utf-8")),
                    "report_estimated_tokens": report_est,
                    "total_token_usage": total_usage,
                    "total_costs": total_cost,
                }
                logger.info(f"TOKEN_USAGE_SNAPSHOT: {json.dumps(snapshot, ensure_ascii=False)}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "token_usage_snapshot",
                        json.dumps(snapshot, ensure_ascii=False),
                        self.researcher.websocket,
                        True,
                        snapshot,
                    )
        except Exception as e:
            logger.debug(f"Failed to log token usage snapshot: {e}")

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "report_written",
                f"📝 Report written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return report

    async def write_report_conclusion(self, report_content: str, research_gap: str = "") -> str:
        """
        Write the conclusion for the report.

        Args:
            report_content (str): The content of the report.
            research_gap (str): Identified research gap.

        Returns:
            str: The generated conclusion.
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_conclusion",
                f"✍️ Writing conclusion for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        conclusion = await write_conclusion(
            query=self.researcher.query,
            context=report_content,
            config=self.researcher.cfg,
            agent_role_prompt=self.researcher.cfg.agent_role or self.researcher.role,
            cost_callback=self.researcher.add_costs,
            websocket=self.researcher.websocket,
            prompt_family=self.researcher.prompt_family,
            research_gap=research_gap,
            **self.researcher.kwargs
        )
        raw_conclusion = conclusion or ""

        # Enforce conclusion language consistency (models sometimes ignore language instructions). :-)
        try:
            conclusion = await self._ensure_output_language(conclusion, self.researcher.cfg.language)
        except Exception:
            pass

        # Post-process conclusion citations to prevent stray URLs (e.g. openai.com) slipping through. :-)
        try:
            ctx = self._coerce_context_to_text(self.researcher.context)
            allowed_for_citations = self._allowed_urls_for_postprocessing(ctx)
            if _env_bool("CITATION_POSTPROCESS", True):
                conclusion = sanitize_citation_links(conclusion, allowed_urls=allowed_for_citations)
                conclusion = canonicalize_intext_citations(conclusion, allowed_urls=allowed_for_citations)
                if _env_bool("STRICT_CITATIONS", True):
                    conclusion = prune_unsupported_citation_claims(conclusion)
        except Exception:
            pass

        try:
            ctx = self._coerce_context_to_text(self.researcher.context)
            allowed_for_citations = self._allowed_urls_for_postprocessing(ctx)
            await self._log_citation_postprocess_debug(
                stage="conclusion",
                before_text=raw_conclusion,
                after_text=conclusion or "",
                allowed_urls=allowed_for_citations,
            )
        except Exception:
            pass

        conclusion = await self._maybe_fill_citations(conclusion)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "conclusion_written",
                f"📝 Conclusion written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return conclusion

    async def write_introduction(self, research_gap: str = ""):
        """Write the introduction section of the report."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_introduction",
                f"✍️ Writing introduction for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        context_text = self._coerce_context_to_text(self.researcher.context)
        introduction = await write_report_introduction(
            query=self.researcher.query,
            context=context_text,
            agent_role_prompt=self.researcher.cfg.agent_role or self.researcher.role,
            config=self.researcher.cfg,
            websocket=self.researcher.websocket,
            cost_callback=self.researcher.add_costs,
            prompt_family=self.researcher.prompt_family,
            research_gap=research_gap,
            **self.researcher.kwargs
        )
        raw_intro = introduction or ""

        # Enforce introduction language consistency (models sometimes ignore language instructions). :-)
        try:
            introduction = await self._ensure_output_language(introduction, self.researcher.cfg.language)
        except Exception:
            pass

        # Post-process introduction citations for consistency with the main report. :-)
        try:
            allowed_for_citations = self._allowed_urls_for_postprocessing(context_text)
            if _env_bool("CITATION_POSTPROCESS", True):
                introduction = sanitize_citation_links(introduction, allowed_urls=allowed_for_citations)
                introduction = canonicalize_intext_citations(introduction, allowed_urls=allowed_for_citations)
                if _env_bool("STRICT_CITATIONS", True):
                    introduction = prune_unsupported_citation_claims(introduction)
        except Exception:
            pass

        await self._log_citation_postprocess_debug(
            stage="introduction",
            before_text=raw_intro,
            after_text=introduction or "",
            allowed_urls=self._allowed_urls_for_postprocessing(context_text),
        )

        introduction = await self._maybe_fill_citations(introduction)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "introduction_written",
                f"📝 Introduction written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return introduction

    async def write_research_gap(self):
        """Write the research gap section of the report."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_research_gap",
                f"🕵️ Writing research gap section for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        from ..actions.report_generation import write_research_gap

        gap_section = await write_research_gap(
            query=self.researcher.query,
            context=self.researcher.context,
            config=self.researcher.cfg,
            websocket=self.researcher.websocket,
            cost_callback=self.researcher.add_costs,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "research_gap_written",
                f"📝 Research gap section written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return gap_section

    async def get_subtopics(self):
        """Retrieve subtopics for the research."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "generating_subtopics",
                f"🌳 Generating subtopics for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        subtopics = await construct_subtopics(
            task=self.researcher.query,
            data=self.researcher.context,
            config=self.researcher.cfg,
            subtopics=self.researcher.subtopics,
            cost_callback=self.researcher.add_costs,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subtopics_generated",
                f"📊 Subtopics generated for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return subtopics

    async def get_draft_section_titles(self, current_subtopic: str):
        """Generate draft section titles for the report."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "generating_draft_sections",
                f"📑 Generating draft section titles for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        draft_section_titles = await generate_draft_section_titles(
            query=self.researcher.query,
            current_subtopic=current_subtopic,
            context=self.researcher.context,
            role=self.researcher.cfg.agent_role or self.researcher.role,
            websocket=self.researcher.websocket,
            config=self.researcher.cfg,
            cost_callback=self.researcher.add_costs,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "draft_sections_generated",
                f"🗂️ Draft section titles generated for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return draft_section_titles
