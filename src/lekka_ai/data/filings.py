"""Filing document cleaning and section splitting for SEC 10-K and 10-Q reports."""

from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup

from lekka_ai.logging import get_logger

logger = get_logger(__name__)

# Regex pattern matching 10-K and 10-Q section item headings
# (e.g. "Item 1", "ITEM 1A", "Part I, Item 1")
ITEM_HEADING_PATTERN = re.compile(
    r"(?i)(?:^|\n)\s*(?:part\s+[iIvV]+[,\s]+)?item\s+([0-9]{1,2}[a-z]?)\b[\.\:\-\s]*([^\n]*)",
    re.MULTILINE,
)

# Standard block elements in HTML that should introduce line breaks in text
BLOCK_TAGS = (
    "p",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "tr",
    "br",
    "li",
    "hr",
    "table",
    "header",
    "footer",
    "section",
    "article",
)


def clean_html_to_text(html_content: str) -> str:
    """Clean raw SEC filing HTML content into normalized text.

    Strips non-content tags (scripts, styles, metadata), replaces HTML entities
    and non-breaking spaces, preserves paragraph and section breaks, and normalizes
    consecutive whitespace.

    Args:
        html_content: Raw HTML text from SEC EDGAR primary document.

    Returns:
        Cleaned, readable plain text string.
    """
    if not html_content or not html_content.strip():
        return ""

    soup = BeautifulSoup(html_content, "lxml")

    # Decompose script, style, and metadata tags
    for tag in soup(["script", "style", "noscript", "meta", "link", "svg", "button"]):
        tag.decompose()

    # Append newlines to block tags to maintain natural paragraph/heading separation
    for tag in soup.find_all(BLOCK_TAGS):
        tag.append("\n")

    text = soup.get_text()

    # Replace non-breaking spaces and special entity representations
    text = text.replace("\xa0", " ").replace("&nbsp;", " ")

    # Normalize horizontal whitespace on each line
    lines: list[str] = []
    for line in text.splitlines():
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)

    # Collapse consecutive blank lines
    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def split_filing_sections(text: str) -> dict[str, str]:
    """Split 10-K or 10-Q text into sections by 'Item' headings.

    Distinguishes actual body content from Table of Contents (TOC) entries by
    evaluating segment length and TOC patterns (e.g. trailing page numbers or dot
    leaders). When an item occurs multiple times (e.g. in TOC and body), the
    substantial body occurrence is selected.

    Args:
        text: Cleaned filing text string.

    Returns:
        Dictionary mapping canonical section names (e.g. 'Item 1', 'Item 1A',
        'Item 7', 'Item 7A', 'Item 8') to section body text.
    """
    if not text or not text.strip():
        return {}

    raw_matches = list(ITEM_HEADING_PATTERN.finditer(text))
    if not raw_matches:
        return {}

    segments: list[dict[str, Any]] = []
    n_matches = len(raw_matches)

    for i, match in enumerate(raw_matches):
        item_num = match.group(1).upper()
        header_text = match.group(0).strip()
        content_start = match.end()
        content_end = raw_matches[i + 1].start() if i + 1 < n_matches else len(text)

        content = text[content_start:content_end].strip()

        # Detect if this match was part of Table of Contents
        is_toc = bool(
            re.search(r"\.{3,}|\bpage\s+\d+|\d+$", header_text, re.IGNORECASE)
            or len(content) < 80
        )

        segments.append(
            {
                "key": f"Item {item_num}",
                "header": header_text,
                "content": content,
                "length": len(content),
                "is_toc": is_toc,
            }
        )

    # Group candidates by section key, selecting the longest content segment
    sections: dict[str, str] = {}
    for seg in segments:
        key = seg["key"]
        content = seg["content"]

        if key not in sections:
            sections[key] = content
        else:
            # Prefer significantly longer segment over TOC or brief snippet
            if len(content) > len(sections[key]):
                sections[key] = content

    return sections


def get_section(
    sections: dict[str, str],
    section_name: str,
) -> str | None:
    """Retrieve section text by name, handling missing sections gracefully.

    Supports normalized queries such as 'Item 1A', 'item 1a', 'ITEM 1A', or '1a'.

    Args:
        sections: Dictionary returned by split_filing_sections.
        section_name: Name or item identifier of the requested section.

    Returns:
        Section text if found, or None if missing.
    """
    clean_name = section_name.strip()

    # Direct match
    if clean_name in sections:
        return sections[clean_name]

    # Normalize: e.g. "item 1a" -> "Item 1A", "1a" -> "Item 1A"
    norm = clean_name.upper()
    if norm.startswith("ITEM"):
        norm_item = norm.replace("ITEM", "").strip()
    else:
        norm_item = norm

    canonical_key = f"Item {norm_item}"
    return sections.get(canonical_key)


def parse_filing_document(
    html_content: str,
) -> tuple[str, dict[str, str]]:
    """Clean raw HTML document and split into sections.

    Returns:
        Tuple of (full_cleaned_text, sections_dict).
    """
    cleaned_text = clean_html_to_text(html_content)
    sections = split_filing_sections(cleaned_text)
    return cleaned_text, sections
