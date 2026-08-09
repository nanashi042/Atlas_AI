"""Gemini-powered analysis for documents extracted by PdfExtractionService.

This service is intentionally Telegram-free and database-free. It receives
an :class:`ExtractedDocument` (already sanitised by the PDF layer), builds a
bounded analysis prompt, and delegates the actual generation to the
existing Gemini wrapper so model fallback and retry behaviour stay
identical to the rest of Atlas AI.

The service:

* Truncates very large extracted texts at safe page boundaries using
  :data:`app.config.settings.DOCUMENT_MAX_CHARACTERS` and
  :data:`app.config.settings.DOCUMENT_MAX_PAGES`.
* Falls back to a deterministic template built only from the structured
  data (filename, page count, available text) when Gemini is unavailable
  so the user always gets a useful response.
* Never logs PDF contents, prompts with document contents, or Telegram
  file paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.ai.llm import generate_response
from app.config.settings import settings
from app.services.documents.document_models import (
    ExtractedDocument,
    ExtractionStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# User-facing messages
# ---------------------------------------------------------------------------


EMPTY_DOCUMENT_MESSAGE = (
    "📄 Document Analysis\n\n"
    "This PDF appears to be empty — it has no pages to read.\n\n"
    "Source: Uploaded document"
)


IMAGE_ONLY_MESSAGE = (
    "📄 Document Analysis\n\n"
    "I couldn't find any extractable text in this PDF.\n\n"
    "It looks like a scanned or image-only document. OCR isn't enabled in "
    "this step of Atlas AI, so I can't summarize its visual contents yet.\n\n"
    "Source: Uploaded document"
)


PARTIAL_DOCUMENT_MESSAGE = (
    "⚠️ I was able to read only part of this PDF; the analysis below is based "
    "on the pages with extractable text."
)


UNEXPECTED_ERROR_MESSAGE = (
    "⚠️ I couldn't analyze this document right now. "
    "Please try again shortly."
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DocumentAnalysisResult:
    """Final user-facing analysis text plus a flag describing truncation."""

    text: str
    truncated: bool = False
    used_fallback: bool = False


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_DOCUMENT_ANALYSIS_PROMPT_TEMPLATE = """You are Atlas AI Financial Copilot analyzing an uploaded document.

The user uploaded a PDF. Its extracted text is provided below. Page numbers
in the source are 1-based. Use ONLY information present in the supplied
text. If a piece of information is not present, say so honestly.

Required sections (omit only when there is truly nothing to say):

� Summary
- 2-4 sentences describing what the document appears to be and its main point.

🔑 Key Points
- Up to 5 bullet points of the most important topics, facts, or conclusions.

📊 Important Numbers
- Up to 5 bullet points highlighting material numbers, dates, or financial
  metrics when they appear. Quote values verbatim. Skip this section entirely
  when the document contains no quantitative information.

⚠️ Notes / Risks
- Up to 3 bullet points of caveats, risks, or follow-up items that the
  document itself supports. Skip this section when there is nothing to add.

Hard rules:
- Never invent facts, numbers, dates, company names, prices, or metrics.
- Distinguish facts (in the document) from interpretation.
- For financial documents, clearly mark anything that is interpretation.
- This is informational analysis only. It is NOT financial advice and is
  NOT a buy/sell/hold recommendation.
- Keep the whole response under ~1200 characters.
- Do not start with "I" or refer to yourself by name.
- Do not include the literal "Source:" line in your output.
- Mobile-friendly formatting only (short lines, no tables).

DOCUMENT METADATA:
- Filename: {filename}
- Page count: {page_count}
- Document title (from metadata): {title}
- Extraction status: {status}
{truncation_notice}
EXTRACTED TEXT:
---
{extracted_text}
---
"""


def _build_analysis_prompt(
    document: ExtractedDocument,
    bounded_text: str,
    truncated: bool,
) -> str:
    """Render the analysis prompt with sanitised metadata only."""
    title = document.title or "Unknown (not set in PDF metadata)"
    truncation_notice = ""
    if truncated:
        truncation_notice = (
            f"- Note: only the first {settings.DOCUMENT_MAX_CHARACTERS} characters "
            "of extracted text are included below (a subset of the full document). "
            "Some pages or page tails may be missing from the analysis.\n"
        )
    return _DOCUMENT_ANALYSIS_PROMPT_TEMPLATE.format(
        filename=document.filename,
        page_count=document.page_count,
        title=title,
        status=document.status.value,
        truncation_notice=truncation_notice,
        extracted_text=bounded_text or "[no extractable text]",
    )


# ---------------------------------------------------------------------------
# Text bounding
# ---------------------------------------------------------------------------


def _bound_document_text(
    document: ExtractedDocument,
) -> tuple[str, bool]:
    """Return (bounded_text, truncated_flag).

    The truncation respects page boundaries: we keep whole pages until adding
    another would exceed the configured character budget. We also drop any
    page beyond :data:`settings.DOCUMENT_MAX_PAGES` so a 1000-page PDF does
    not stall extraction.
    """
    max_chars = settings.DOCUMENT_MAX_CHARACTERS
    max_pages = settings.DOCUMENT_MAX_PAGES

    kept_pages = []
    total = 0
    truncated = False

    for page in document.pages:
        if page.page_number > max_pages:
            truncated = True
            break
        candidate = page.text or ""
        # +2 accounts for the "\n\n" join we'll add between pages.
        if kept_pages and total + len(candidate) + 2 > max_chars:
            truncated = True
            break
        if not kept_pages and len(candidate) > max_chars:
            kept_pages.append(candidate[:max_chars])
            total = max_chars
            truncated = True
            break
        kept_pages.append(candidate)
        total += len(candidate)
        if total > max_chars:
            truncated = True
            break

    return "\n\n".join(p for p in kept_pages if p), truncated


# ---------------------------------------------------------------------------
# Fallback template
# ---------------------------------------------------------------------------


def _build_fallback_text(document: ExtractedDocument, truncated: bool) -> str:
    """Deterministic, structured summary built only from extracted text."""
    lines = ["📄 Document Analysis", ""]
    title = document.title or "Unknown"
    lines.append(f"Title: {title}")
    lines.append(f"File: {document.filename}")
    lines.append(f"Pages: {document.page_count}")
    lines.append("")

    text = (document.extracted_text or "").strip()
    if not text:
        lines.append(
            "No extractable text was found in this PDF. It may be scanned or "
            "image-only. OCR isn't enabled in this step of Atlas AI."
        )
        lines.append("")
        lines.append("Source: Uploaded document")
        return "\n".join(lines)

    summary = text[:600].strip()
    if len(text) > 600:
        # Stop on a word boundary when possible for a cleaner cut.
        cut = summary.rfind(" ")
        if cut > 200:
            summary = summary[:cut] + "…"
        else:
            summary = summary + "…"

    lines.append("📝 Summary")
    lines.append(summary)
    lines.append("")
    lines.append("🔑 Key Points")
    # Surface up to 3 short candidate sentences as bullets (deterministic
    # fallback; Gemini will replace these with curated points when available).
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 30]
    for sentence in sentences[:3]:
        lines.append(f"• {sentence}.")
    if len(sentences) < 1:
        lines.append("• No additional structured points available.")

    lines.append("")
    lines.append("Source: Uploaded document")
    if truncated:
        lines.append("")
        lines.append("⚠️ This analysis is based on a subset of the document.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DocumentAnalysisError(Exception):
    """Raised by DocumentAnalysisService for unexpected failures."""


class DocumentAnalysisService:
    """Synthesises a concise AI analysis of an extracted PDF document.

    The service delegates to :func:`app.ai.llm.generate_response` so the
    Gemini client, model fallback chain, and logging behaviour are shared
    with the rest of Atlas AI.
    """

    def __init__(self, llm_generate=None):
        # Allow dependency injection for tests.
        self.llm_generate = llm_generate

    async def analyze(self, document: ExtractedDocument) -> DocumentAnalysisResult:
        """Return a user-facing analysis string for the supplied document.

        Handles every extraction status produced by ``PdfExtractionService``:

        * ``SUCCESS``  -> LLM-generated analysis.
        * ``PARTIAL``  -> LLM-generated analysis with a truncation notice.
        * ``EMPTY_DOCUMENT`` / ``NO_EXTRACTABLE_TEXT`` -> structured
          fallback message; the LLM is NOT called.
        """
        if not isinstance(document, ExtractedDocument):
            raise DocumentAnalysisError(
                "analyze() requires an ExtractedDocument instance."
            )

        if document.status == ExtractionStatus.EMPTY_DOCUMENT:
            return DocumentAnalysisResult(text=EMPTY_DOCUMENT_MESSAGE)
        if document.status == ExtractionStatus.NO_EXTRACTABLE_TEXT:
            return DocumentAnalysisResult(text=IMAGE_ONLY_MESSAGE)

        bounded_text, truncated = _bound_document_text(document)
        if truncated:
            logger.info(
                "Document analysis input truncated (pages=%d, max_chars=%d).",
                document.page_count,
                settings.DOCUMENT_MAX_CHARACTERS,
            )

        try:
            prompt = _build_analysis_prompt(document, bounded_text, truncated)
            response_text = await self._call_llm(prompt)
        except Exception as exc:
            logger.error("Document analysis Gemini call failed: %s", exc)
            response_text = ""

        if not response_text or response_text.startswith("[Error") or response_text.startswith("⚠️"):
            logger.warning("Document analysis falling back to deterministic template.")
            text = _build_fallback_text(document, truncated)
            return DocumentAnalysisResult(text=text, truncated=truncated, used_fallback=True)

        title = document.title or "Unknown"
        header = f"📄 Document Analysis\n\nTitle: {title}\nPages: {document.page_count}"
        text = response_text.strip()
        if document.status == ExtractionStatus.PARTIAL:
            text = f"{PARTIAL_DOCUMENT_MESSAGE}\n\n{text}" if text else PARTIAL_DOCUMENT_MESSAGE
        text = f"{header}\n\n{text}"
        if "Source: Uploaded document" not in text:
            text = f"{text.rstrip()}\n\nSource: Uploaded document"

        return DocumentAnalysisResult(text=text, truncated=truncated, used_fallback=False)

    async def _call_llm(self, prompt: str) -> str:
        if self.llm_generate is not None:
            return await self.llm_generate(prompt, history=[])
        return await generate_response(prompt, history=[])


# Module-level singleton — handlers import this directly.
document_analysis_service = DocumentAnalysisService()
