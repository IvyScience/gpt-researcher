import re
import markdown
from typing import List, Dict


def extract_headers(markdown_text: str) -> List[Dict]:
    """
    Extract headers from markdown text.

    Args:
        markdown_text (str): The markdown text to process.

    Returns:
        List[Dict]: A list of dictionaries representing the header structure.
    """
    headers = []
    parsed_md = markdown.markdown(markdown_text)
    lines = parsed_md.split("\n")

    stack = []
    for line in lines:
        if line.startswith("<h") and len(line) > 2 and line[2].isdigit():
            level = int(line[2])
            header_text = line[line.index(">") + 1 : line.rindex("<")]

            while stack and stack[-1]["level"] >= level:
                stack.pop()

            header = {
                "level": level,
                "text": header_text,
            }
            if stack:
                stack[-1].setdefault("children", []).append(header)
            else:
                headers.append(header)

            stack.append(header)

    return headers


def extract_sections(markdown_text: str) -> List[Dict[str, str]]:
    """
    Extract all written sections from subtopic report.

    Args:
        markdown_text (str): Subtopic report text.

    Returns:
        List[Dict[str, str]]: List of sections, each section is a dictionary containing
        'section_title' and 'written_content'.
    """
    sections = []
    parsed_md = markdown.markdown(markdown_text)

    pattern = r'<h\d>(.*?)</h\d>(.*?)(?=<h\d>|$)'
    matches = re.findall(pattern, parsed_md, re.DOTALL)

    for title, content in matches:
        clean_content = re.sub(r'<.*?>', '', content).strip()
        if clean_content:
            sections.append({
                "section_title": title.strip(),
                "written_content": clean_content
            })

    return sections


def table_of_contents(markdown_text: str) -> str:
    """
    Generate a table of contents for the given markdown text.

    Args:
        markdown_text (str): The markdown text to process.

    Returns:
        str: The generated table of contents.
    """
    def generate_table_of_contents(headers, indent_level=0):
        toc = ""
        for header in headers:
            toc += " " * (indent_level * 4) + "- " + header["text"] + "\n"
            if "children" in header:
                toc += generate_table_of_contents(header["children"], indent_level + 1)
        return toc

    try:
        headers = extract_headers(markdown_text)
        toc = "## Table of Contents\n\n" + generate_table_of_contents(headers)
        return toc
    except Exception as e:
        print("table_of_contents Exception : ", e)
        return markdown_text


def add_references(report_markdown: str, visited_urls: set) -> str:
    """
    Add references to the markdown report.

    Args:
        report_markdown (str): The existing markdown report.
        visited_urls (set): A set of URLs that have been visited during research.

    Returns:
        str: The updated markdown report with added references.
    """
    try:
        url_markdown = "\n\n\n## References\n\n"
        url_markdown += "".join(f"- [{url}]({url})\n" for url in visited_urls)
        updated_markdown_report = report_markdown + url_markdown
        return updated_markdown_report
    except Exception as e:
        print(f"Encountered exception in adding source urls : {e}")
        return report_markdown


def sanitize_citation_links(report_markdown: str, allowed_urls: set[str] | None = None) -> str:
    """
    Remove placeholder / fake citation links (e.g. example.com, '(url)') while keeping the citation text.

    This protects against LLMs emitting template placeholders like:
      ([Author, 2024](https://example.com))
      ([in-text citation](url))
    """
    try:
        import re
        from urllib.parse import urlparse

        # Match standard markdown links, but skip images: ![alt](url)
        pattern = re.compile(r"(?<!!)\[(?P<text>[^\]]+)\]\((?P<url>[^)]+)\)")

        def _is_placeholder(u: str) -> bool:
            u = (u or "").strip().strip('"').strip("'")
            if not u:
                return True
            if u.lower() == "url":
                return True
            if "exampleurl" in u.lower():
                return True
            if "example.com" in u.lower():
                return True
            if "example.org" in u.lower():
                return True
            if "openai.com" in u.lower():
                return True
            return False

        def _normalize_url(u: str) -> str:
            u = (u or "").strip().strip('"').strip("'")
            return u

        def _canonical_for_allowlist(u: str) -> str:
            """
            Canonicalize URLs for allowlist comparisons (fixes false negatives like http/https, trailing slash). :-)
            """
            u = _normalize_url(u)
            if not u:
                return ""
            try:
                parsed = urlparse(u)
                scheme = (parsed.scheme or "").lower()
                netloc = (parsed.netloc or "").lower()
                path = (parsed.path or "").rstrip("/")
                # Normalize doi.org scheme/host
                if netloc in {"doi.org", "dx.doi.org"}:
                    scheme = "https"
                    netloc = "doi.org"
                if scheme and netloc:
                    return f"{scheme}://{netloc}{path}"
            except Exception:
                pass
            return u.rstrip("/")

        allowed_canon: set[str] | None = None
        if allowed_urls is not None:
            allowed_canon = {_canonical_for_allowlist(a) for a in allowed_urls if a}

        def repl(m: re.Match) -> str:
            text = m.group("text")
            url = m.group("url")
            url_clean = _normalize_url(url)

            # Fast path for obvious placeholders
            if _is_placeholder(url_clean):
                return text

            # If we have an allow-list of sources, drop any citation link not in the allow-list.
            # This prevents hallucinated domains from polluting the report.
            if allowed_urls is not None:
                if allowed_canon is not None and _canonical_for_allowlist(url_clean) not in allowed_canon:
                    return text

            # Some models emit malformed "url" tokens; treat non-URLs as placeholders in citations
            try:
                parsed = urlparse(url_clean)
                if parsed.scheme and parsed.netloc:
                    return m.group(0)
            except Exception:
                pass

            # If it doesn't look like a URL, drop the link but keep the label
            return text

        return pattern.sub(repl, report_markdown)
    except Exception:
        return report_markdown


def canonicalize_intext_citations(report_markdown: str, allowed_urls: set[str] | None = None) -> str:
    """
    Force in-text citations to the canonical parenthetical markdown-link form: ([label](url))

    Only transforms citations that are already in a parenthetical markdown-link form:
      ([Anything](url))
    Leaves reference lists and other links untouched.
    """
    try:
        import re
        from urllib.parse import urlparse

        # Parenthetical markdown link: ([label](url))
        # Allow empty label/url so we can normalize cases like:
        #   ([](https://...))  or  ([Smith, 2023]())
        pattern = re.compile(r"\(\s*\[(?P<label>[^\]]*)\]\((?P<url>[^)]*)\)\s*\)")

        def _canonical_for_allowlist(u: str) -> str:
            u = (u or "").strip().strip('"').strip("'")
            if not u:
                return ""
            try:
                parsed = urlparse(u)
                scheme = (parsed.scheme or "").lower()
                netloc = (parsed.netloc or "").lower()
                path = (parsed.path or "").rstrip("/")
                if netloc in {"doi.org", "dx.doi.org"}:
                    scheme = "https"
                    netloc = "doi.org"
                if scheme and netloc:
                    return f"{scheme}://{netloc}{path}"
            except Exception:
                pass
            return u.rstrip("/")

        allowed_canon: set[str] | None = None
        if allowed_urls is not None:
            allowed_canon = {_canonical_for_allowlist(a) for a in allowed_urls if a}

        def repl(m: re.Match) -> str:
            label = (m.group("label") or "").strip()
            url = (m.group("url") or "").strip().strip('"').strip("'")
            safe_label = label or "Citation"

            if not url:
                return f"({safe_label})"
            if allowed_canon is not None and _canonical_for_allowlist(url) not in allowed_canon:
                # If it's not an allowed source, drop the link but keep the label
                return f"({safe_label})"
            return f"([{safe_label}]({url}))"

        return pattern.sub(repl, report_markdown)
    except Exception:
        return report_markdown


def extract_source_urls_from_context(context: str) -> set[str]:
    """
    Extract source URLs/IDs from the context string that is provided to the LLM.
    Supports both default and Granite prompt-family formats. :-)
    """
    try:
        import re

        if not context:
            return set()

        sources: set[str] = set()

        # Default prompt family: "Source: <url>"
        for m in re.finditer(r"(?mi)^\s*Source:\s*(?P<url>\S+)\s*$", context):
            u = m.group("url").strip()
            if u and u.lower() not in ("none", "null", "n/a"):
                sources.add(u)

        # Granite 3.x: "Document <document_id>"
        for m in re.finditer(r"(?mi)^\s*Document\s+(?P<url>\S+)\s*$", context):
            sources.add(m.group("url").strip())

        # Granite 3.3: document {"document_id": "..."}
        for m in re.finditer(r'"document_id"\s*:\s*"(?P<url>[^"]+)"', context):
            sources.add(m.group("url").strip())

        return sources
    except Exception:
        return set()


def prune_unsupported_citation_claims(report_markdown: str) -> str:
    """
    Prune common hallucinated "study/meta-analysis found X%" style claims when they are not cited.

    This is intentionally conservative: it only drops sentences that look like
    research-claim statements AND contain no markdown links (i.e. no citations).

    Use this as a safety net; the primary defense should be prompt constraints + good context.
    """
    try:
        import re

        def _extract_context_sources(text: str) -> set[str]:
            sources: set[str] = set()
            if not text:
                return sources

            def _valid(u: str) -> bool:
                return u and u.lower() not in ("none", "null", "n/a")

            # Default prompt family: "Source: <url>"
            for m in re.finditer(r"(?mi)^\s*Source:\s*(?P<url>\S+)\s*$", text):
                u = m.group("url").strip()
                if _valid(u):
                    sources.add(u)
            # Granite prompt family: "Document <document_id>"
            for m in re.finditer(r"(?mi)^\s*Document\s+(?P<url>\S+)\s*$", text):
                u = m.group("url").strip()
                if _valid(u):
                    sources.add(u)
            # Granite 3.3 format: document {"document_id": "..."}
            for m in re.finditer(r'"document_id"\s*:\s*"(?P<url>[^"]+)"', text):
                u = m.group("url").strip()
                if _valid(u):
                    sources.add(u)
            return sources

        # Heuristics: sentences that are likely to be fabricated without sources
        risky = re.compile(
            r"\b("
            r"study|studies|meta-analysis|systematic review|randomi[sz]ed|longitudinal|cohort|trial|"
            r"found that|revealed that|demonstrated that|showed that|reported a|were \d+(\.\d+)?x|"
            r"\d+(\.\d+)?%|times more likely|statistically significant"
            r")\b",
            re.IGNORECASE,
        )

        # Chinese heuristics: keep conservative (only strong "research says" phrases) :-)
        risky_zh = re.compile(
            r"(研究表明|研究发现|文献表明|文献显示|有研究指出|实证研究表明|调查发现|数据表明|数据显示)"
        )

        # Split by paragraph to keep markdown structure
        paragraphs = report_markdown.split("\n\n")
        out_paras: list[str] = []
        for p in paragraphs:
            stripped = p.strip()
            if not stripped:
                out_paras.append(p)
                continue

            # Keep headers, lists, code blocks as-is
            if stripped.startswith(("#", "- ", "* ", "```")):
                out_paras.append(p)
                continue

            # Sentence-ish split (simple, language-agnostic enough for our use case)
            parts = re.split(r"(?<=[.!?。！？])\s+", p)
            kept: list[str] = []
            for s in parts:
                s_strip = s.strip()
                if not s_strip:
                    continue
                has_citation_link = "](" in s_strip  # markdown link
                if (not has_citation_link) and (risky.search(s_strip) or risky_zh.search(s_strip)):
                    # Drop uncited risky claim
                    continue
                kept.append(s)

            out_paras.append(" ".join(kept).strip())

        return "\n\n".join([p for p in out_paras if p is not None])
    except Exception:
        return report_markdown
