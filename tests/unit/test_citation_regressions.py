def test_visited_urls_are_copied_on_init():
    """visited_urls must not be shared/mutated across researchers. :-)"""
    from gpt_researcher.agent import GPTResearcher

    shared = {"https://a.com"}
    r = GPTResearcher(query="test", visited_urls=shared)

    assert r.visited_urls is not shared

    shared.add("https://b.com")
    assert "https://b.com" not in r.visited_urls


def test_allowlist_canonicalization_preserves_doi_variants():
    """Allowlist should not strip DOI link variants (http/https, dx.doi.org, trailing slash). :-)"""
    from gpt_researcher.actions.markdown_processing import (
        canonicalize_intext_citations,
        sanitize_citation_links,
    )

    allowed = {"https://doi.org/10.1234/ABC"}
    md = "Claim **([X, 2024](http://dx.doi.org/10.1234/ABC/))**."

    out = sanitize_citation_links(md, allowed_urls=allowed)
    assert out == md

    out2 = canonicalize_intext_citations(out, allowed_urls=allowed)
    assert out2 == md


def test_placeholder_domains_are_always_stripped_even_if_allowed():
    """Placeholder domains should always be stripped as safety net. :-)"""
    from gpt_researcher.actions.markdown_processing import sanitize_citation_links

    md = "x **([OpenAI, n.d.](https://www.openai.com))**."
    out = sanitize_citation_links(md, allowed_urls={"https://www.openai.com"})
    assert out == "x **(OpenAI, n.d.)**."

    md2 = "y **([Source](https://example.org))**."
    out2 = sanitize_citation_links(md2, allowed_urls={"https://example.org"})
    assert out2 == "y **(Source)**."

    md3 = "z **([Source](https://www.exampleurl1.com))**."
    out3 = sanitize_citation_links(md3, allowed_urls={"https://www.exampleurl1.com"})
    assert out3 == "z **(Source)**."


def test_subtopic_prompt_does_not_encourage_placeholder_urls():
    """Subtopic prompt should not include example.org/openai.com or bare (Source) guidance. :-)"""
    from gpt_researcher.prompts import PromptFamily

    p = PromptFamily.generate_subtopic_report_prompt(
        current_subtopic="x",
        existing_headers=[],
        relevant_written_contents=[],
        main_topic="y",
        context="Source: https://doi.org/10.1/abc\nTitle: t\nContent: c\n",
        report_format="apa",
        language="chinese",
    )
    # It's OK to *mention* placeholder domains as forbidden, but we should never embed them as example links.
    assert "https://example.org" not in p
    assert "https://example.com" not in p
    assert "https://www.openai.com" not in p
    # Should explicitly forbid bare Source without URL
    assert "NEVER output bare `(Source)`" in p or "NEVER output bare (Source)" in p
