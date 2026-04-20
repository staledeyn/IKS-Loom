"""PDF text extraction and cleaning for digital PDFs."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

# Lines that look like page headers/footers (stray page numbers)
_PAGE_NUMBER_LINE = re.compile(r"^\d{1,4}$")
_PAGE_WORD_LINE = re.compile(r"^page\s+\d+\s*$", re.IGNORECASE)
# Excessive blank lines -> at most double newline
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
# Repeated horizontal whitespace within a line
_EXCESS_INLINE_SPACE = re.compile(r"[ \t]{2,}")
# Zero-width and invisible formatting characters
_ZERO_WIDTH = re.compile(r"[\u200b-\u200d\ufeff]")
# Soft hyphen (common PDF artifact)
_SOFT_HYPHEN = "\u00ad"


def _validate_doc_id(doc_id: str) -> str:
    if not doc_id or not doc_id.strip():
        raise ValueError("doc_id must be a non-empty string.")
    safe = doc_id.strip()
    if re.search(r"[./\\]", safe):
        raise ValueError("doc_id must not contain path separators or '.' segments.")
    return safe


def _strip_non_printable_artifacts(text: str) -> str:
    """Remove control characters and non-printable artifacts; keep newlines and tabs."""
    out: list[str] = []
    for ch in text:
        if ch in "\n\t":
            out.append(ch)
            continue
        if ch == _SOFT_HYPHEN:
            continue
        if ord(ch) < 32 or ord(ch) == 127:
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            continue
        if ch.isprintable():
            out.append(ch)
    return "".join(out)


def _remove_stray_page_numbers(text: str) -> str:
    """Drop lines that are only small integers or 'Page N' (typical headers/footers)."""
    lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if _PAGE_NUMBER_LINE.match(stripped):
            continue
        if _PAGE_WORD_LINE.match(stripped):
            continue
        lines.append(line)
    return "\n".join(lines)


def _normalize_and_clean_text(raw: str) -> str:
    """Apply regex and normalization to produce stable, readable plain text."""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    text = _strip_non_printable_artifacts(text)
    text = _remove_stray_page_numbers(text)
    text = _EXCESS_INLINE_SPACE.sub(" ", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def process_pdf(file_path: str, doc_id: str) -> Path:
    """
    Open a digital PDF, extract text page by page, clean it, and save to processed storage.

    Args:
        file_path: Absolute or relative path to the source PDF.
        doc_id: Identifier used for the output filename (no path separators).

    Returns:
        Path to the written ``{doc_id}.txt`` file under ``data/processed``.

    Raises:
        ValueError: If ``doc_id`` is invalid.
        FileNotFoundError: If the PDF does not exist.
        RuntimeError: If the PDF cannot be read or text normalization fails.
        OSError: If the output file cannot be written.
    """
    safe_id = _validate_doc_id(doc_id)
    pdf_path = Path(file_path).expanduser().resolve()
    out_path = (_PROCESSED_DIR / safe_id).with_suffix(".txt")

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    page_texts: list[str] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    raw = page.extract_text()
                except Exception as page_exc:
                    logger.warning("Failed to extract text from page %s of %s: %s", i + 1, pdf_path, page_exc)
                    continue
                if raw is None:
                    logger.debug("No text layer on page %s of %s", i + 1, pdf_path)
                    continue
                page_texts.append(raw)
    except FileNotFoundError:
        raise
    except Exception as exc:
        logger.exception("Failed to open or read PDF: %s", pdf_path)
        raise RuntimeError(f"Could not process PDF: {pdf_path}") from exc

    combined = "\n\n".join(page_texts)

    try:
        cleaned = _normalize_and_clean_text(combined)
    except Exception as exc:
        logger.exception("Text normalization failed for doc_id=%s", safe_id)
        raise RuntimeError("Failed to normalize extracted text.") from exc

    try:
        out_path.write_text(cleaned, encoding="utf-8", newline="\n")
    except OSError as exc:
        logger.error("Could not write output file %s: %s", out_path, exc)
        raise

    logger.info("Wrote cleaned text for doc_id=%s to %s", safe_id, out_path)
    return out_path
