"""
graph/document_parser.py — file-type routing and text extraction.

Why a separate module (not inline in the node)?
  The node function needs to stay thin so LangGraph can reason about it cleanly.
  All the messy I/O — temp-file handling, loader selection, encoding fallbacks —
  lives here.  This also makes the parsers unit-testable without spinning up a
  LangGraph graph.

Supported formats and the loader used for each:
  .pdf  → LangChain PyPDFLoader   (uses pypdf under the hood)
  .docx → LangChain Docx2txtLoader (uses docx2txt under the hood)
  .txt  → plain Python open() read  (no LangChain needed — it's just text)
  .eml  → Python's built-in email library  (LangChain has no native .eml loader)
  text  → caller already has the string; nothing to load

Why not OCR?
  Out of scope per the project spec.  If a scanned-image PDF is uploaded, pypdf
  will return empty text.  We surface that as an informative error rather than
  silently producing garbage extractions.

Temp-file handling:
  FastAPI gives us an UploadFile whose bytes we read into memory.  For PyPDFLoader
  and Docx2txtLoader we need a path on disk, so we write to a NamedTemporaryFile,
  load, then delete.  We use delete=False + explicit cleanup in a finally block
  because Windows does not allow two handles to the same temp file simultaneously
  (which would happen with delete=True inside a with-block).
"""

from __future__ import annotations

import email
import logging
import os
import tempfile
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────

def extract_text(
    *,
    file_bytes: bytes | None = None,
    file_name: str | None = None,
    file_type: str | None = None,
    raw_text: str | None = None,
) -> str:
    """
    Return the plain-text content of a complaint document.

    Callers pass EITHER (file_bytes + file_type) OR raw_text.
    If raw_text is provided it is returned immediately — no parsing needed.

    Parameters
    ----------
    file_bytes : raw bytes of the uploaded file
    file_name  : original filename (used only for logging)
    file_type  : one of "pdf" | "docx" | "txt" | "eml" | "text"
    raw_text   : pre-extracted text (paste input — skips all loaders)

    Raises
    ------
    ValueError  : unsupported file type or empty extraction result
    """
    # Fast path — caller already has the text (paste input)
    if raw_text:
        logger.info("extract_text: using pre-supplied raw_text (%d chars)", len(raw_text))
        return raw_text.strip()

    if not file_bytes:
        raise ValueError("extract_text: no file_bytes and no raw_text provided")

    resolved_type = (file_type or "").lower().strip(".")
    logger.info(
        "extract_text: parsing '%s' as type='%s' (%d bytes)",
        file_name or "unknown",
        resolved_type,
        len(file_bytes),
    )

    if resolved_type == "pdf":
        return _parse_pdf(file_bytes, file_name)
    elif resolved_type == "docx":
        return _parse_docx(file_bytes, file_name)
    elif resolved_type == "txt":
        return _parse_txt(file_bytes)
    elif resolved_type == "eml":
        return _parse_eml(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type '{resolved_type}'. "
            "Accepted: pdf, docx, txt, eml."
        )


# ── Format-specific parsers ───────────────────────────────────────────────────

def _parse_pdf(file_bytes: bytes, file_name: str | None) -> str:
    """
    Extract text page-by-page from PDF bytes.

    Uses in-memory pypdf.PdfReader first for zero disk-I/O and complete Windows
    safety (avoids file locking issues and NamedTemporaryFile permission errors).
    Falls back to LangChain's PyPDFLoader if needed.
    """
    import io
    import unicodedata

    # 1. High-speed in-memory extraction via pypdf (recommended & Windows-safe)
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text: list[str] = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                pages_text.append(t.strip())

        text = "\n\n".join(pages_text)
        if text.strip():
            # Normalize non-breaking hyphens and unusual unicode that breaks on Windows terminals
            text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
            logger.info("_parse_pdf: in-memory extracted %d chars from %d pages for '%s'", len(text), len(reader.pages), file_name or "document")
            return text
    except Exception as mem_exc:
        logger.warning("_parse_pdf: in-memory parse failed (%s), trying PyPDFLoader fallback", mem_exc)

    # 2. Fallback: LangChain PyPDFLoader via temporary file with safe Windows cleanup
    from langchain_community.document_loaders import PyPDFLoader
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        if not pages:
            raise ValueError(
                f"PDF loader returned no pages for '{file_name}'. "
                "The file may be empty or contain only scanned images (OCR not supported)."
            )

        text = "\n\n".join(p.page_content.strip() for p in pages if p.page_content.strip())
        if not text:
            raise ValueError(
                f"PDF '{file_name}' produced no extractable text. "
                "It may be a scanned image-only PDF (OCR not supported)."
            )

        text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        logger.info("_parse_pdf: fallback extracted %d chars from %d pages", len(text), len(pages))
        return text

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _parse_docx(file_bytes: bytes, file_name: str | None) -> str:
    """
    Use LangChain's Docx2txtLoader to extract text from a .docx file.

    Same temp-file pattern as _parse_pdf — the loader needs a path on disk.
    docx2txt preserves paragraph breaks but strips formatting (bold, tables, etc.)
    which is exactly what we want: clean prose for the LLM.
    """
    from langchain_community.document_loaders import Docx2txtLoader

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = Docx2txtLoader(tmp_path)
        docs = loader.load()   # typically returns a single Document for a .docx

        text = "\n\n".join(d.page_content.strip() for d in docs if d.page_content.strip())

        if not text:
            raise ValueError(f"Docx2txtLoader extracted no text from '{file_name}'.")

        logger.info("_parse_docx: extracted %d chars", len(text))
        return text

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _parse_txt(file_bytes: bytes) -> str:
    """
    Decode a plain-text file.

    We try UTF-8 first (the safe modern default), then fall back to latin-1.
    latin-1 never raises a UnicodeDecodeError because every byte is a valid
    latin-1 character — it's the universal last-resort decoder for legacy files
    from older QMS systems that may have been exported without a BOM.
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("_parse_txt: UTF-8 decode failed, retrying with latin-1")
        text = file_bytes.decode("latin-1")

    text = text.strip()
    if not text:
        raise ValueError("The uploaded .txt file appears to be empty.")

    logger.info("_parse_txt: extracted %d chars", len(text))
    return text


def _parse_eml(file_bytes: bytes) -> str:
    """
    Parse a raw .eml file using Python's standard-library email module.

    LangChain has no native .eml loader, so we use the built-in `email`
    package directly.  We walk all MIME parts and collect:
      - Subject, From, To, Date headers (metadata the LLM can use)
      - text/plain parts (the actual email body)
    We skip text/html parts to avoid sending HTML tags to the LLM, and
    skip attachments entirely (out of scope for Phase 2).

    Why not just decode the raw bytes as text?
      Raw .eml files contain MIME headers, quoted-printable or base64 encoding,
      and multi-part boundaries that would confuse the LLM if passed verbatim.
      Parsing them properly gives the LLM clean, structured prose.
    """
    msg = email.message_from_bytes(file_bytes)

    # Build a header block so the LLM has metadata context
    headers = []
    for header_name in ("Subject", "From", "To", "Date"):
        value = msg.get(header_name, "")
        if value:
            headers.append(f"{header_name}: {value}")

    # Walk MIME parts and collect plain-text bodies
    body_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode(charset, errors="replace"))
    else:
        # Non-multipart — the payload IS the body
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            body_parts.append(payload.decode(charset, errors="replace"))

    body = "\n\n".join(p.strip() for p in body_parts if p.strip())

    if not body:
        raise ValueError("Could not extract any plain-text body from the .eml file.")

    text = "\n".join(headers) + "\n\n" + body
    logger.info("_parse_eml: extracted %d chars", len(text))
    return text
