"""Telegram handler for uploaded PDF documents.

This module is the only place that knows how to combine a Telegram
``Update`` containing a PDF document with the rest of Atlas AI:

* download the file to a secure temporary location,
* run it through the existing :class:`PdfExtractionService`,
* ask the existing :class:`DocumentAnalysisService` for an analysis,
* send the result back to the user in mobile-friendly chunks,
* always clean up the temporary file (including on error).

The handler never logs PDF contents, document text, or filesystem paths.
It returns plain user-facing strings; Telegram-specific formatting is
limited to Markdown-style newlines and section emojis.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Optional

from telegram import Update
from telegram.ext import ContextTypes

from app.config.settings import settings
from app.services.documents.document_analysis_service import (
    DocumentAnalysisError,
    DocumentAnalysisResult,
    DocumentAnalysisService,
    EMPTY_DOCUMENT_MESSAGE,
    IMAGE_ONLY_MESSAGE,
    UNEXPECTED_ERROR_MESSAGE,
)
from app.services.documents.document_models import (
    ExtractedDocument,
    ExtractionStatus,
)
from app.services.documents.pdf_service import (
    InvalidPdfError,
    PasswordProtectedPdfError,
    PdfExtractionService,
    PdfFileNotFoundError,
)
from app.services.documents.document_qa_service import document_context_store

logger = logging.getLogger(__name__)


# Telegram has a 4096 char limit per message; we leave a safety margin.
_TELEGRAM_MESSAGE_LIMIT = max(500, int(getattr(settings, "TELEGRAM_MAX_MESSAGE_LENGTH", 3900)))


# User-facing error messages. They never expose filesystem paths or
# document contents; they only tell the user what category of problem
# occurred.
_UNSUPPORTED_FILE_MESSAGE = (
    "⚠️ Only PDF files are supported. Please send the document as a PDF."
)

_DOWNLOAD_FAILURE_MESSAGE = (
    "⚠️ I couldn't download that file from Telegram. "
    "Please try uploading it again."
)

_PASSWORD_PROTECTED_MESSAGE = (
    "⚠️ This PDF is password-protected. "
    "Please remove the password and re-upload it."
)

_INVALID_PDF_MESSAGE = (
    "⚠️ This file doesn't look like a readable PDF. "
    "It may be corrupted or use an unsupported format."
)

_MISSING_FILE_MESSAGE = (
    "⚠️ I couldn't read the uploaded file. "
    "Please try uploading it again."
)

_PARTIAL_EXTRACTION_NOTE = (
    "I was only able to read part of this PDF — the analysis below is "
    "based on the pages I could extract."
)

_DOWNLOAD_PROGRESS_MESSAGE = "📄 Analyzing your PDF..."


class TelegramPdfHandler:
    """Encapsulates the download -> extract -> analyze -> reply flow.

    All external dependencies (the Telegram bot, the PDF service, the
    analysis service, and the message-length limit) are injectable so
    unit tests can drive every code path without making real API calls.
    """

    def __init__(
        self,
        pdf_service: Optional[PdfExtractionService] = None,
        analysis_service: Optional[DocumentAnalysisService] = None,
        telegram_message_limit: int = _TELEGRAM_MESSAGE_LIMIT,
        document_context_store_instance=None,
    ):
        self.pdf_service = pdf_service or PdfExtractionService()
        self.analysis_service = analysis_service or DocumentAnalysisService()
        self.telegram_message_limit = telegram_message_limit
        self.document_context_store = document_context_store_instance or document_context_store

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def handle_pdf(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Telegram document handler entry point.

        Catches every error locally and always replies with a user-friendly
        message. The bot never crashes because of an uploaded PDF.
        """
        message = getattr(update, "message", None)
        if message is None:
            logger.warning("PDF handler received update without a message; ignoring.")
            return

        document = getattr(message, "document", None)
        if document is None:
            await self._reply(message, _UNSUPPORTED_FILE_MESSAGE)
            return

        if not self._is_pdf(document):
            await self._reply(message, _UNSUPPORTED_FILE_MESSAGE)
            return

        await self._reply(message, _DOWNLOAD_PROGRESS_MESSAGE)

        temp_path: Optional[Path] = None
        try:
            temp_path = self._safe_temp_path(document.file_name)
            try:
                await self._download_to(message, document, temp_path)
            except RuntimeError as exc:
                logger.warning("Telegram PDF download failed: %s", exc)
                await self._reply(message, _DOWNLOAD_FAILURE_MESSAGE)
                return

            try:
                extracted = self.pdf_service.extract(temp_path)
            except PdfFileNotFoundError:
                logger.warning("PDF handler: extracted file vanished before parsing.")
                await self._reply(message, _MISSING_FILE_MESSAGE)
                return
            except PasswordProtectedPdfError:
                await self._reply(message, _PASSWORD_PROTECTED_MESSAGE)
                return
            except InvalidPdfError:
                await self._reply(message, _INVALID_PDF_MESSAGE)
                return

            # Extraction reads from an anonymous temporary filename. Preserve
            # the user's original basename in the in-memory model so the
            # analysis can identify the uploaded document without retaining it.
            extracted = replace(
                extracted,
                filename=os.path.basename(getattr(document, "file_name", "") or "document.pdf"),
            )

            # These states do not have usable text, so do not spend a Gemini
            # request (and do not require an analysis-service implementation
            # to know Telegram's delivery concerns).
            if extracted.status == ExtractionStatus.EMPTY_DOCUMENT:
                await self._reply(message, EMPTY_DOCUMENT_MESSAGE)
                return
            if extracted.status == ExtractionStatus.NO_EXTRACTABLE_TEXT:
                await self._reply(message, IMAGE_ONLY_MESSAGE)
                return

            # Retain only a bounded, in-memory extracted context. The source
            # PDF itself is removed by the ``finally`` block below.
            user = getattr(update, "effective_user", None)
            user_id = str(getattr(user, "id", "default"))
            self.document_context_store.set_document(user_id, extracted)

            try:
                result = await self.analysis_service.analyze(extracted)
            except DocumentAnalysisError as exc:
                logger.error("Document analysis failed: %s", exc)
                await self._reply(message, UNEXPECTED_ERROR_MESSAGE)
                return
            except Exception as exc:  # defensive: never crash the bot
                logger.error("Unexpected document analysis error: %s", exc, exc_info=True)
                await self._reply(message, UNEXPECTED_ERROR_MESSAGE)
                return

            await self._reply_with_result(message, extracted, result)
        except Exception as exc:
            # Final safety net — log the category, never the document text.
            logger.error("PDF handler fatal error: %s", exc, exc_info=True)
            await self._reply(message, UNEXPECTED_ERROR_MESSAGE)
        finally:
            self._cleanup(temp_path)

    # ------------------------------------------------------------------
    # Reply helpers
    # ------------------------------------------------------------------

    async def _reply(self, message, text: str) -> None:
        """Best-effort Telegram send that swallows delivery errors."""
        try:
            await message.reply_text(text)
        except Exception as exc:
            logger.warning("Telegram reply failed: %s", exc)

    async def _reply_with_result(
        self,
        message,
        extracted: ExtractedDocument,
        result: DocumentAnalysisResult,
    ) -> None:
        text = result.text
        if result.truncated and extracted.status != ExtractionStatus.PARTIAL:
            text = self._append_truncation_note(text)
        chunks = self.split_for_telegram(text)
        for chunk in chunks:
            await self._reply(message, chunk)

    def _append_truncation_note(self, text: str) -> str:
        note = (
            "⚠️ This analysis is based on a subset of the document "
            "(large or truncated PDF)."
        )
        if note in text:
            return text
        if "Source: Uploaded document" in text:
            return text.replace(
                "Source: Uploaded document",
                f"{note}\n\nSource: Uploaded document",
            )
        return f"{text.rstrip()}\n\n{note}"

    def split_for_telegram(self, text: str) -> list[str]:
        """Split a long reply into Telegram-safe chunks.

        Splits on blank lines where possible so we never break a paragraph
        across messages. If a single paragraph is larger than the limit,
        we fall back to a hard character split on that block.
        """
        if not text:
            return [""]
        if len(text) <= self.telegram_message_limit:
            return [text]

        chunks: list[str] = []
        remaining = text
        while len(remaining) > self.telegram_message_limit:
            # Prefer a paragraph or line break within the current Telegram
            # window. Include the separator in the sent chunk so joining the
            # chunks is lossless and no whitespace is silently invented.
            window = remaining[: self.telegram_message_limit]
            split_at = window.rfind("\n\n")
            if split_at >= 0:
                split_at += 2
            else:
                split_at = window.rfind("\n")
                if split_at >= 0:
                    split_at += 1
            if split_at <= 0:
                # An oversized single line/word has no safe semantic boundary.
                split_at = self.telegram_message_limit
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:]
        if remaining:
            chunks.append(remaining)
        return chunks

    # ------------------------------------------------------------------
    # Telegram-specific helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_pdf(document) -> bool:
        """Return True when a Telegram Document message is a PDF.

        The Telegram API exposes both ``mime_type`` and the file name.
        We check both because either may be missing depending on how the
        user uploaded the file.
        """
        mime = (getattr(document, "mime_type", "") or "").lower()
        if mime == "application/pdf":
            return True
        file_name = (getattr(document, "file_name", "") or "").lower()
        return file_name.endswith(".pdf")

    @staticmethod
    def _safe_temp_path(file_name: Optional[str]) -> Path:
        """Return a unique, writable temp file path for the download."""
        # Strip any directory components the Telegram client may have sent.
        suffix = ".pdf"
        if file_name:
            base = os.path.basename(file_name)
            _, ext = os.path.splitext(base)
            if ext.lower() == ".pdf":
                suffix = ".pdf"
        fd, name = tempfile.mkstemp(prefix="atlas_pdf_", suffix=suffix)
        os.close(fd)
        return Path(name)

    @staticmethod
    async def _download_to(message, document, destination: Path) -> None:
        """Download a Telegram Document to a local path.

        Raises ``RuntimeError`` if Telegram does not deliver the file —
        the outer handler turns that into a user-friendly message.
        """
        try:
            tg_file = await document.get_file()
            await tg_file.download_to_drive(custom_path=str(destination))
        except Exception as exc:
            raise RuntimeError(f"telegram download failed: {exc}") from exc

    @staticmethod
    def _cleanup(path: Optional[Path]) -> None:
        """Best-effort removal of the temporary file."""
        if path is None:
            return
        try:
            if path.is_file():
                path.unlink()
        except OSError as exc:
            logger.warning("Temporary PDF cleanup failed: %s", exc)


# Module-level singleton — telegram_bot.py imports this directly.
telegram_pdf_handler = TelegramPdfHandler()


async def handle_pdf_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Top-level handler entry point suitable for ``MessageHandler``."""
    await telegram_pdf_handler.handle_pdf(update, context)
