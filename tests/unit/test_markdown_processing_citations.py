import os
import sys


# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from gpt_researcher.actions.markdown_processing import (  # noqa: E402
    canonicalize_intext_citations,
    sanitize_citation_links,
)


def test_sanitize_citation_links_drops_placeholder_url_token():
    md = "This is a claim ([Smith, 2023](url))."
    out = sanitize_citation_links(md, allowed_urls=None)
    assert out == "This is a claim (Smith, 2023)."


def test_sanitize_citation_links_drops_example_com():
    md = "This is a claim ([Smith, 2023](https://example.com/paper))."
    out = sanitize_citation_links(md, allowed_urls=None)
    assert out == "This is a claim (Smith, 2023)."


def test_sanitize_citation_links_keeps_allowed_urls():
    md = "This is a claim ([Smith, 2023](https://a.com/paper))."
    out = sanitize_citation_links(md, allowed_urls={"https://a.com/paper"})
    assert out == md


def test_sanitize_citation_links_drops_disallowed_urls_but_keeps_label():
    md = "This is a claim ([Smith, 2023](https://a.com/paper))."
    out = sanitize_citation_links(md, allowed_urls={"https://b.com/other"})
    assert out == "This is a claim (Smith, 2023)."


def test_canonicalize_intext_citations_preserves_label():
    md = "This is a claim ([Smith, 2023](https://a.com/paper))."
    out = canonicalize_intext_citations(md, allowed_urls=None)
    assert out == md


def test_canonicalize_intext_citations_strips_link_if_disallowed_url():
    md = "This is a claim ([Smith, 2023](https://a.com/paper))."
    out = canonicalize_intext_citations(md, allowed_urls={"https://b.com/other"})
    assert out == "This is a claim (Smith, 2023)."


def test_canonicalize_intext_citations_returns_label_only_if_url_empty():
    md = "This is a claim ([Smith, 2023]())."
    out = canonicalize_intext_citations(md, allowed_urls=None)
    assert out == "This is a claim (Smith, 2023)."


def test_canonicalize_intext_citations_defaults_label_when_missing():
    md = "This is a claim ([](https://a.com/paper))."
    out = canonicalize_intext_citations(md, allowed_urls=None)
    assert out == "This is a claim ([Citation](https://a.com/paper))."


def test_canonicalize_intext_citations_only_transforms_parenthetical_links():
    md = "See [Smith, 2023](https://a.com/paper) for details."
    out = canonicalize_intext_citations(md, allowed_urls=None)
    assert out == md
