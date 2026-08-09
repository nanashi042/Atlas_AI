"""In-memory conversational Q&A over a user's active uploaded document."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Optional

from app.ai.llm import generate_response
from app.config.settings import settings
from app.services.documents.document_models import ExtractedDocument, ExtractedPage

logger = logging.getLogger(__name__)

NO_ACTIVE_DOCUMENT_MESSAGE = "I don't have an active document. Upload a PDF first."
DOCUMENT_CONTEXT_CLEARED_MESSAGE = "Document context cleared."
DOCUMENT_QA_UNAVAILABLE_MESSAGE = (
    "⚠️ I couldn't answer a question about the uploaded document right now. "
    "Please try again shortly."
)


@dataclass(frozen=True)
class ActiveDocumentContext:
    """A bounded, non-persistent document available for one user session."""

    document: ExtractedDocument
    truncated: bool


def _bound_document(document: ExtractedDocument) -> ActiveDocumentContext:
    """Create the page-preserving context permitted by existing limits."""
    pages: list[ExtractedPage] = []
    total = 0
    truncated = False
    for page in document.pages:
        if page.page_number > settings.DOCUMENT_MAX_PAGES:
            truncated = True
            break
        text = page.text or ""
        separator_size = 2 if pages else 0
        if total + separator_size + len(text) <= settings.DOCUMENT_MAX_CHARACTERS:
            pages.append(page)
            total += separator_size + len(text)
            continue
        if not pages and text:
            pages.append(replace(page, text=text[:settings.DOCUMENT_MAX_CHARACTERS], character_count=settings.DOCUMENT_MAX_CHARACTERS))
        truncated = True
        break

    bounded = replace(
        document,
        pages=pages,
        extracted_text="\n\n".join(page.text for page in pages if page.text),
    )
    return ActiveDocumentContext(document=bounded, truncated=truncated)


class DocumentContextStore:
    """Small in-memory active-document registry keyed by Telegram user ID."""

    def __init__(self):
        self._contexts: dict[str, ActiveDocumentContext] = {}

    def set_document(self, user_id: str, document: ExtractedDocument) -> ActiveDocumentContext:
        context = _bound_document(document)
        self._contexts[str(user_id)] = context
        logger.info("Active document context updated (pages=%d, truncated=%s).", document.page_count, context.truncated)
        return context

    def get_document(self, user_id: str) -> Optional[ActiveDocumentContext]:
        return self._contexts.get(str(user_id))

    def clear_document(self, user_id: str) -> bool:
        return self._contexts.pop(str(user_id), None) is not None


def is_document_clear_request(message: str) -> bool:
    return bool(re.search(r"\b(?:forget|clear|remove|delete)\b.*\b(?:this )?(?:document|pdf|report)\b", (message or "").lower()))


def is_explicit_document_question(message: str) -> bool:
    return bool(re.search(r"\b(?:this|the|uploaded|my)\s+(?:document|pdf|report|section)\b|\baccording to (?:the )?(?:document|pdf|report)\b", (message or "").lower()))


def is_likely_document_question(message: str) -> bool:
    """Conservative natural-language detector used only with active context."""
    text = (message or "").lower().strip()
    if is_explicit_document_question(text):
        return True
    if not re.match(r"^(?:what|when|where|who|which|how|did|does|were|was|is|are)\b", text):
        return False
    subjects = ("revenue", "risk", "risks", "r&d", "research and development", "income", "profit", "loss", "debt", "cash flow", "guidance", "metric", "metrics", "financial", "date", "report", "page")
    return any(term in text for term in subjects) or text.startswith("how much")


_DOCUMENT_QA_PROMPT = """You are answering a question about an uploaded document.
Use ONLY the supplied document content. Do not use outside knowledge.
If the answer is not supported by the document, say exactly that you could not find it in the uploaded document. Never invent facts or page numbers.
Preserve numbers, dates, names, and units accurately. Clearly distinguish a direct document fact from interpretation. Keep the answer concise and mobile friendly. When the supporting page marker is available, finish with "Source: Page N" (or relevant page numbers). Do not add a heading.

QUESTION:
{question}

DOCUMENT CONTEXT:
{truncation_notice}
{pages}
"""


def _render_pages(document: ExtractedDocument) -> str:
    return "\n\n".join(f"[Page {page.page_number}]\n{page.text}" for page in document.pages if page.text) or "[No extractable text]"


class DocumentQaService:
    """Answers a question using only the supplied active-document context."""

    def __init__(self, llm_generate=None):
        self.llm_generate = llm_generate

    async def answer(self, question: str, context: ActiveDocumentContext) -> str:
        if not isinstance(context, ActiveDocumentContext):
            raise TypeError("context must be an ActiveDocumentContext")
        prompt = _DOCUMENT_QA_PROMPT.format(
            question=question,
            truncation_notice=("Important: this is a bounded extract, so do not claim unavailable pages were checked." if context.truncated else ""),
            pages=_render_pages(context.document),
        )
        try:
            response = await self._call_llm(prompt)
        except Exception as exc:
            logger.error("Document Q&A Gemini call failed: %s", exc)
            return DOCUMENT_QA_UNAVAILABLE_MESSAGE
        response = (response or "").strip()
        if not response or response.startswith("[Error") or response.startswith("⚠️"):
            logger.warning("Document Q&A received no usable Gemini response.")
            return DOCUMENT_QA_UNAVAILABLE_MESSAGE
        text = f"📄 From the uploaded document:\n\n{response}"
        if context.truncated:
            text += "\n\n⚠️ This answer is based on the available bounded extract."
        return text

    async def _call_llm(self, prompt: str) -> str:
        if self.llm_generate is not None:
            return await self.llm_generate(prompt, history=[])
        return await generate_response(prompt, history=[])


document_context_store = DocumentContextStore()
document_qa_service = DocumentQaService()
