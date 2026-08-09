"""Unit tests for the Telegram PDF upload handler.

The handler is exercised end-to-end but every external dependency is
mocked:

* The Telegram ``Update``/``Document`` objects are lightweight fakes.
* Telegram downloads are intercepted so no real HTTP calls are made.
* The Gemini call goes through a fake ``DocumentAnalysisService`` so the
  test can deterministically choose between SUCCESS / PARTIAL / EMPTY /
  NO_EXTRACTABLE_TEXT and Gemini failure.

Tests construct real PDFs on disk with PyMuPDF so we exercise the same
extraction code path used in production.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fitz

from app.bot.pdf_handler import TelegramPdfHandler, _UNSUPPORTED_FILE_MESSAGE
from app.services.documents.document_analysis_service import (
    DocumentAnalysisResult,
    DocumentAnalysisService,
    EMPTY_DOCUMENT_MESSAGE,
    IMAGE_ONLY_MESSAGE,
    UNEXPECTED_ERROR_MESSAGE,
)
from app.services.documents.document_models import (
    ExtractedDocument,
    ExtractedPage,
    ExtractionStatus,
)
from app.services.documents.document_qa_service import DocumentContextStore


# ---------------------------------------------------------------------------
# Telegram Update / Document fakes
# ---------------------------------------------------------------------------


class FakeDocument:
    def __init__(self, *, mime_type="application/pdf", file_name="report.pdf", file_id="abc"):
        self.mime_type = mime_type
        self.file_name = file_name
        self.file_id = file_id

        self._download_target: Path | None = None
        self._fail_download = False

    def schedule_failure(self):
        self._fail_download = True

    async def get_file(self):
        if self._fail_download:
            raise RuntimeError("telegram download failed")
        fake_file = MagicMock()
        fake_file.download_to_drive = AsyncMock(
            side_effect=self._simulate_download
        )
        return fake_file

    async def _simulate_download(self, custom_path: str):
        # Write the bytes the PDF service will then read.
        target = Path(custom_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self._download_target.read_bytes() if self._download_target else b"")


class FakeMessage:
    def __init__(self, document: FakeDocument | None = None):
        self.document = document
        self.replies: list[str] = []
        self._reply_exc: Exception | None = None

    def schedule_reply_failure(self, exc: Exception):
        self._reply_exc = exc

    async def reply_text(self, text: str, *args, **kwargs):
        if self._reply_exc is not None:
            raise self._reply_exc
        self.replies.append(text)


class FakeUpdate:
    def __init__(self, message: FakeMessage):
        self.message = message


# ---------------------------------------------------------------------------
# Analysis service fake
# ---------------------------------------------------------------------------


class FakeAnalysisService:
    """Drop-in replacement for DocumentAnalysisService with canned output."""

    def __init__(self, result: DocumentAnalysisResult | None = None, raise_exc: Exception | None = None):
        self.result = result or DocumentAnalysisResult(text="ANALYSIS_RESULT")
        self.raise_exc = raise_exc
        self.calls: list[ExtractedDocument] = []

    async def analyze(self, document: ExtractedDocument) -> DocumentAnalysisResult:
        self.calls.append(document)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.result


# ---------------------------------------------------------------------------
# PDF fixture helpers
# ---------------------------------------------------------------------------


def _make_pdf(path: Path, pages: list[str], title: str | None = None) -> None:
    doc = fitz.open()
    if title:
        metadata = doc.metadata
        metadata["title"] = title
        doc.set_metadata(metadata)
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPdfHandlerRouting(unittest.IsolatedAsyncioTestCase):
    async def test_non_pdf_document_is_rejected_with_friendly_message(self):
        handler = TelegramPdfHandler(analysis_service=FakeAnalysisService())
        message = FakeMessage(FakeDocument(mime_type="text/plain", file_name="notes.txt"))
        await handler.handle_pdf(FakeUpdate(message), MagicMock())
        self.assertEqual(len(message.replies), 1)
        self.assertIn("Only PDF files are supported", message.replies[0])

    async def test_pdf_with_pdf_extension_but_wrong_mime_is_accepted(self):
        # Telegram sometimes sends an empty mime_type for documents; we
        # accept anything ending in .pdf.
        tmp = tempfile.TemporaryDirectory()
        try:
            directory = Path(tmp.name)
            pdf_path = directory / "annual.pdf"
            _make_pdf(pdf_path, ["Revenue was $100M."])
            doc = FakeDocument(mime_type="", file_name="annual.pdf")
            doc._download_target = pdf_path

            analysis = FakeAnalysisService(DocumentAnalysisResult(text="ANALYSIS_RESULT"))
            handler = TelegramPdfHandler(analysis_service=analysis)
            message = FakeMessage(doc)
            await handler.handle_pdf(FakeUpdate(message), MagicMock())

            # Progress message + analysis message
            self.assertGreaterEqual(len(message.replies), 2)
            self.assertEqual(message.replies[-1], "ANALYSIS_RESULT")
            self.assertEqual(len(analysis.calls), 1)
        finally:
            tmp.cleanup()

    async def test_update_without_message_is_ignored(self):
        handler = TelegramPdfHandler(analysis_service=FakeAnalysisService())
        # Update without message attribute at all.
        class _Empty:
            message = None
        await handler.handle_pdf(_Empty(), MagicMock())


class TestSuccessfulFlow(unittest.IsolatedAsyncioTestCase):
    async def test_successful_pdf_creates_active_document_context(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            directory = Path(tmp.name)
            pdf_path = directory / "report.pdf"
            _make_pdf(pdf_path, ["Revenue was $100M."])
            store = DocumentContextStore()
            doc = FakeDocument(mime_type="application/pdf", file_name="report.pdf")
            doc._download_target = pdf_path
            handler = TelegramPdfHandler(
                analysis_service=FakeAnalysisService(),
                document_context_store_instance=store,
            )
            await handler.handle_pdf(FakeUpdate(FakeMessage(doc)), MagicMock())
            context = store.get_document("default")
            self.assertIsNotNone(context)
            self.assertEqual(context.document.filename, "report.pdf")
            self.assertIn("Revenue was $100M.", context.document.extracted_text)
        finally:
            tmp.cleanup()

    async def test_full_flow_downloads_extracts_and_replies(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            directory = Path(tmp.name)
            pdf_path = directory / "report.pdf"
            _make_pdf(pdf_path, ["Revenue was $100M.", "Net income was $20M."], title="Annual Report")

            doc = FakeDocument(mime_type="application/pdf", file_name="report.pdf")
            doc._download_target = pdf_path
            analysis = FakeAnalysisService(DocumentAnalysisResult(text="Summary text here."))
            handler = TelegramPdfHandler(analysis_service=analysis)
            message = FakeMessage(doc)

            await handler.handle_pdf(FakeUpdate(message), MagicMock())

            # Progress message + analysis reply
            self.assertGreaterEqual(len(message.replies), 2)
            self.assertIn("Analyzing your PDF", message.replies[0])
            self.assertIn("Summary text here.", message.replies[-1])

            # The analysis service received an ExtractedDocument.
            self.assertEqual(len(analysis.calls), 1)
            received = analysis.calls[0]
            self.assertEqual(received.filename, "report.pdf")
            self.assertEqual(received.page_count, 2)
            self.assertEqual(received.title, "Annual Report")
        finally:
            tmp.cleanup()

    async def test_temp_file_is_removed_after_success(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            directory = Path(tmp.name)
            pdf_path = directory / "report.pdf"
            _make_pdf(pdf_path, ["Some text."])

            doc = FakeDocument(mime_type="application/pdf", file_name="report.pdf")
            doc._download_target = pdf_path
            handler = TelegramPdfHandler(analysis_service=FakeAnalysisService())
            message = FakeMessage(doc)

            await handler.handle_pdf(FakeUpdate(message), MagicMock())

            # The handler uses a global temp directory; sweep any atlas_*.pdf
            # leftovers from the test run.
            leftovers = [
                p for p in Path(tempfile.gettempdir()).glob("atlas_pdf_*.pdf")
            ]
            self.assertEqual(leftovers, [], f"Temp files not cleaned up: {leftovers}")
        finally:
            tmp.cleanup()


class TestExtractionFailures(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_pdf_replies_with_invalid_message(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            directory = Path(tmp.name)
            bad = directory / "bad.pdf"
            bad.write_bytes(b"not a valid pdf")

            doc = FakeDocument(mime_type="application/pdf", file_name="bad.pdf")
            doc._download_target = bad
            handler = TelegramPdfHandler(analysis_service=FakeAnalysisService())
            message = FakeMessage(doc)
            await handler.handle_pdf(FakeUpdate(message), MagicMock())

            self.assertTrue(any("doesn't look like a readable PDF" in r for r in message.replies))
        finally:
            tmp.cleanup()

    async def test_empty_pdf_replies_with_empty_message_without_calling_llm(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            directory = Path(tmp.name)
            empty = directory / "empty.pdf"
            # Minimal zero-page PDF (PyMuPDF refuses to save one directly).
            empty.write_bytes(
                b"%PDF-1.4\n"
                b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
                b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
                b"xref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000054 00000 n \n"
                b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n98\n%%EOF\n"
            )

            doc = FakeDocument(mime_type="application/pdf", file_name="empty.pdf")
            doc._download_target = empty
            analysis = FakeAnalysisService()
            handler = TelegramPdfHandler(analysis_service=analysis)
            message = FakeMessage(doc)
            await handler.handle_pdf(FakeUpdate(message), MagicMock())

            self.assertTrue(any("appears to be empty" in r for r in message.replies))
            self.assertEqual(analysis.calls, [])
        finally:
            tmp.cleanup()

    async def test_image_only_pdf_replies_with_image_only_message(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            directory = Path(tmp.name)
            pdf = directory / "scan.pdf"
            _make_pdf(pdf, [""])  # one page with no text

            doc = FakeDocument(mime_type="application/pdf", file_name="scan.pdf")
            doc._download_target = pdf
            analysis = FakeAnalysisService()
            handler = TelegramPdfHandler(analysis_service=analysis)
            message = FakeMessage(doc)
            await handler.handle_pdf(FakeUpdate(message), MagicMock())

            self.assertTrue(any("scanned or image-only" in r for r in message.replies))
            self.assertEqual(analysis.calls, [])
        finally:
            tmp.cleanup()

    async def test_password_protected_pdf_replies_with_password_message(self):
        # Build a PDF, then encrypt it via PyMuPDF so it requires a password.
        tmp = tempfile.TemporaryDirectory()
        try:
            directory = Path(tmp.name)
            pdf = directory / "locked.pdf"
            _make_pdf(pdf, ["secret"])

            doc_obj = fitz.open(pdf)
            doc_obj.save(
                directory / "encrypted.pdf",
                encryption=fitz.PDF_ENCRYPT_AES_256,
                owner_pw="owner-secret",
                user_pw="user-secret",
            )
            doc_obj.close()

            doc = FakeDocument(mime_type="application/pdf", file_name="locked.pdf")
            doc._download_target = directory / "encrypted.pdf"
            analysis = FakeAnalysisService()
            handler = TelegramPdfHandler(analysis_service=analysis)
            message = FakeMessage(doc)
            await handler.handle_pdf(FakeUpdate(message), MagicMock())

            # PyMuPDF needs a password to open; the handler should report a
            # password problem without crashing.
            joined = "\n".join(message.replies)
            self.assertIn("password", joined.lower())
            self.assertEqual(analysis.calls, [])
        finally:
            tmp.cleanup()


class TestGeminiFailure(unittest.IsolatedAsyncioTestCase):
    async def test_analysis_exception_returns_user_friendly_error(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            directory = Path(tmp.name)
            pdf = directory / "report.pdf"
            _make_pdf(pdf, ["Hello."])

            doc = FakeDocument(mime_type="application/pdf", file_name="report.pdf")
            doc._download_target = pdf
            analysis = FakeAnalysisService(
                raise_exc=RuntimeError("gemini down")
            )
            handler = TelegramPdfHandler(analysis_service=analysis)
            message = FakeMessage(doc)
            await handler.handle_pdf(FakeUpdate(message), MagicMock())

            joined = "\n".join(message.replies)
            self.assertIn(UNEXPECTED_ERROR_MESSAGE, joined)
        finally:
            tmp.cleanup()


class TestTelegramFailureIsolation(unittest.IsolatedAsyncioTestCase):
    async def test_telegram_send_failure_does_not_crash(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            directory = Path(tmp.name)
            pdf = directory / "report.pdf"
            _make_pdf(pdf, ["Hello."])

            doc = FakeDocument(mime_type="application/pdf", file_name="report.pdf")
            doc._download_target = pdf
            analysis = FakeAnalysisService(DocumentAnalysisResult(text="OK"))
            handler = TelegramPdfHandler(analysis_service=analysis)
            message = FakeMessage(doc)
            message.schedule_reply_failure(RuntimeError("telegram timeout"))
            # The handler must swallow the send failure rather than raise.
            await handler.handle_pdf(FakeUpdate(message), MagicMock())
        finally:
            tmp.cleanup()

    async def test_download_failure_replies_with_friendly_message(self):
        doc = FakeDocument(mime_type="application/pdf", file_name="report.pdf")
        doc.schedule_failure()
        handler = TelegramPdfHandler(analysis_service=FakeAnalysisService())
        message = FakeMessage(doc)
        await handler.handle_pdf(FakeUpdate(message), MagicMock())
        joined = "\n".join(message.replies)
        self.assertIn("couldn't download", joined)


class TestLargeDocumentProtection(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_text_is_truncated_for_llm_prompt(self):
        from app.config.settings import settings

        tmp = tempfile.TemporaryDirectory()
        try:
            directory = Path(tmp.name)
            big = "Atlas revenue was $12345. " * 5000  # very large
            pdf = directory / "huge.pdf"
            _make_pdf(pdf, [big])

            doc = FakeDocument(mime_type="application/pdf", file_name="huge.pdf")
            doc._download_target = pdf

            captured: dict = {}

            class CapturingAnalysis(FakeAnalysisService):
                async def analyze(self, document):
                    captured["doc"] = document
                    return await super().analyze(document)

            handler = TelegramPdfHandler(analysis_service=CapturingAnalysis())
            message = FakeMessage(doc)
            await handler.handle_pdf(FakeUpdate(message), MagicMock())

            self.assertIn("doc", captured)
            # The bounded text inside the extracted document (built by the
            # service's own _bound_document_text) is at most the configured cap.
            self.assertLessEqual(
                len(captured["doc"].extracted_text),
                settings.DOCUMENT_MAX_CHARACTERS + 100,  # slack for joiners
            )
        finally:
            tmp.cleanup()


class TestMessageSplitting(unittest.TestCase):
    def test_short_text_returns_single_chunk(self):
        handler = TelegramPdfHandler(analysis_service=FakeAnalysisService())
        chunks = handler.split_for_telegram("hello world")
        self.assertEqual(chunks, ["hello world"])

    def test_long_text_is_split_on_blank_lines(self):
        handler = TelegramPdfHandler(
            analysis_service=FakeAnalysisService(),
            telegram_message_limit=50,
        )
        text = "Paragraph one is short.\n\n" + "Paragraph two " * 20 + "\n\n" + "Paragraph three."
        chunks = handler.split_for_telegram(text)
        # All chunks must be <= 50 chars and the content reassembles
        # exactly with the original blank-line separator.
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 50)
        self.assertEqual("".join(chunks), text)

    def test_oversized_paragraph_is_hard_split(self):
        handler = TelegramPdfHandler(
            analysis_service=FakeAnalysisService(),
            telegram_message_limit=20,
        )
        big = "x" * 75
        chunks = handler.split_for_telegram(big)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 20)
        self.assertEqual("".join(chunks), big)

    def test_empty_text_returns_single_empty_chunk(self):
        handler = TelegramPdfHandler(analysis_service=FakeAnalysisService())
        self.assertEqual(handler.split_for_telegram(""), [""])


class TestCleanupAlwaysHappens(unittest.IsolatedAsyncioTestCase):
    async def test_temp_file_cleaned_up_on_extraction_error(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            directory = Path(tmp.name)
            bad = directory / "bad.pdf"
            bad.write_bytes(b"garbage")

            doc = FakeDocument(mime_type="application/pdf", file_name="bad.pdf")
            doc._download_target = bad
            handler = TelegramPdfHandler(analysis_service=FakeAnalysisService())
            message = FakeMessage(doc)

            await handler.handle_pdf(FakeUpdate(message), MagicMock())

            leftovers = list(Path(tempfile.gettempdir()).glob("atlas_pdf_*.pdf"))
            self.assertEqual(leftovers, [], f"Leftover temp files: {leftovers}")
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
