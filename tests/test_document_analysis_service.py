"""Unit tests for the Gemini-powered document analysis service.

All Gemini calls are mocked. No real network, no real PDF files required.
"""

import logging
import unittest
from unittest.mock import AsyncMock

from app.services.documents.document_analysis_service import (
    DocumentAnalysisError,
    DocumentAnalysisResult,
    DocumentAnalysisService,
    EMPTY_DOCUMENT_MESSAGE,
    IMAGE_ONLY_MESSAGE,
    PARTIAL_DOCUMENT_MESSAGE,
    UNEXPECTED_ERROR_MESSAGE,
    _bound_document_text,
    _build_fallback_text,
    _build_analysis_prompt,
)
from app.services.documents.document_models import (
    ExtractedDocument,
    ExtractedPage,
    ExtractionStatus,
)


def _make_page(page_number: int, text: str) -> ExtractedPage:
    return ExtractedPage(page_number=page_number, text=text, character_count=len(text))


def _make_document(
    pages,
    *,
    filename: str = "report.pdf",
    title=None,
    status=ExtractionStatus.SUCCESS,
):
    extracted_text = "\n\n".join(p.text for p in pages if p.text)
    return ExtractedDocument(
        filename=filename,
        page_count=len(pages),
        extracted_text=extracted_text,
        pages=list(pages),
        status=status,
        title=title,
        warnings=[],
    )


class FakeLlm:
    """Captures every prompt and returns a configured response."""

    def __init__(self, response: str = "LLM analysis text.", raise_exc: Exception | None = None):
        self.response = response
        self.raise_exc = raise_exc
        self.calls: list[str] = []

    async def __call__(self, prompt: str, history=None):
        self.calls.append(prompt)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


class TestBounding(unittest.TestCase):
    def test_short_document_is_not_truncated(self):
        doc = _make_document([_make_page(1, "Hello world.")])
        text, truncated = _bound_document_text(doc)
        self.assertEqual(text, "Hello world.")
        self.assertFalse(truncated)

    def test_page_boundary_is_respected_when_truncating(self):
        from app.config.settings import settings
        # Two pages; the second would push us past the cap.
        long_a = "a" * (settings.DOCUMENT_MAX_CHARACTERS - 100)
        long_b = "b" * 500
        doc = _make_document([_make_page(1, long_a), _make_page(2, long_b)])
        text, truncated = _bound_document_text(doc)
        self.assertTrue(truncated)
        self.assertIn("a", text)
        # The second page must be excluded entirely — no "b" should survive.
        self.assertNotIn("b", text)

    def test_pages_above_max_pages_are_skipped(self):
        from app.config.settings import settings
        pages = [_make_page(i, f"page {i}") for i in range(1, settings.DOCUMENT_MAX_PAGES + 5)]
        doc = _make_document(pages)
        text, truncated = _bound_document_text(doc)
        self.assertTrue(truncated)
        # The very last configured page must be present; pages past the cap
        # must be excluded.
        self.assertIn(f"page {settings.DOCUMENT_MAX_PAGES}", text)
        self.assertNotIn(f"page {settings.DOCUMENT_MAX_PAGES + 1}", text)

    def test_single_oversized_page_is_hard_truncated(self):
        from app.config.settings import settings
        huge = "x" * (settings.DOCUMENT_MAX_CHARACTERS + 500)
        doc = _make_document([_make_page(1, huge)])
        text, truncated = _bound_document_text(doc)
        self.assertEqual(len(text), settings.DOCUMENT_MAX_CHARACTERS)
        self.assertTrue(truncated)


class TestPromptConstruction(unittest.TestCase):
    def test_prompt_includes_metadata_and_text(self):
        doc = _make_document(
            [_make_page(1, "Revenue was $100M.")],
            filename="q1.pdf",
            title="Q1 2026 Report",
        )
        prompt = _build_analysis_prompt(doc, "Revenue was $100M.", truncated=False)
        self.assertIn("q1.pdf", prompt)
        self.assertIn("Q1 2026 Report", prompt)
        self.assertIn("Revenue was $100M.", prompt)
        # Must contain the safety contract.
        self.assertIn("Never invent", prompt)
        self.assertIn("NOT financial advice", prompt)
        # Must NOT echo the user-facing footer line into the prompt itself.
        self.assertNotIn("Source: Uploaded document", prompt)

    def test_truncation_notice_is_included_when_truncated(self):
        doc = _make_document([_make_page(1, "some text")])
        prompt = _build_analysis_prompt(doc, "some text", truncated=True)
        self.assertIn("subset", prompt.lower())


class TestEmptyAndImageOnly(unittest.IsolatedAsyncioTestCase):
    async def test_empty_document_short_circuits_without_calling_llm(self):
        llm = FakeLlm(response="SHOULD NOT BE CALLED")
        service = DocumentAnalysisService(llm_generate=llm)
        doc = _make_document([], status=ExtractionStatus.EMPTY_DOCUMENT)
        result = await service.analyze(doc)
        self.assertIsInstance(result, DocumentAnalysisResult)
        self.assertIn(EMPTY_DOCUMENT_MESSAGE, result.text)
        self.assertEqual(llm.calls, [])

    async def test_image_only_short_circuits_without_calling_llm(self):
        llm = FakeLlm(response="SHOULD NOT BE CALLED")
        service = DocumentAnalysisService(llm_generate=llm)
        doc = _make_document(
            [_make_page(1, "")], status=ExtractionStatus.NO_EXTRACTABLE_TEXT
        )
        result = await service.analyze(doc)
        self.assertIn(IMAGE_ONLY_MESSAGE, result.text)
        self.assertEqual(llm.calls, [])


class TestSuccessfulAnalysis(unittest.IsolatedAsyncioTestCase):
    async def test_llm_response_is_returned_with_source_footer(self):
        llm = FakeLlm(response="Here is the analysis.")
        service = DocumentAnalysisService(llm_generate=llm)
        doc = _make_document(
            [_make_page(1, "Revenue was $100M.")],
            title="Annual Report",
        )
        result = await service.analyze(doc)
        self.assertEqual(
            result.text,
            "📄 Document Analysis\n\nTitle: Annual Report\nPages: 1\n\n"
            "Here is the analysis.\n\nSource: Uploaded document",
        )
        self.assertFalse(result.used_fallback)
        self.assertFalse(result.truncated)
        self.assertEqual(len(llm.calls), 1)

    async def test_partial_status_prepends_partial_notice(self):
        llm = FakeLlm(response="Analysis OK.")
        service = DocumentAnalysisService(llm_generate=llm)
        doc = _make_document(
            [_make_page(1, "Partial content.")],
            status=ExtractionStatus.PARTIAL,
        )
        result = await service.analyze(doc)
        self.assertIn(PARTIAL_DOCUMENT_MESSAGE, result.text)
        self.assertIn("Analysis OK.", result.text)
        self.assertIn("Source: Uploaded document", result.text)

    async def test_invalid_input_raises_document_analysis_error(self):
        service = DocumentAnalysisService(llm_generate=FakeLlm())
        with self.assertRaises(DocumentAnalysisError):
            await service.analyze("not a document")  # type: ignore[arg-type]


class TestGeminiFailure(unittest.IsolatedAsyncioTestCase):
    async def test_error_string_response_uses_fallback(self):
        llm = FakeLlm(response="[Error 503] Gemini service unavailable")
        service = DocumentAnalysisService(llm_generate=llm)
        doc = _make_document([_make_page(1, "Hello world.")])
        result = await service.analyze(doc)
        self.assertTrue(result.used_fallback)
        self.assertIn("Source: Uploaded document", result.text)

    async def test_warning_emoji_response_uses_fallback(self):
        llm = FakeLlm(response="⚠️ Gemini service is currently busy.")
        service = DocumentAnalysisService(llm_generate=llm)
        doc = _make_document([_make_page(1, "Hello world.")])
        result = await service.analyze(doc)
        self.assertTrue(result.used_fallback)

    async def test_raised_exception_uses_fallback(self):
        llm = FakeLlm(raise_exc=RuntimeError("boom"))
        service = DocumentAnalysisService(llm_generate=llm)
        doc = _make_document([_make_page(1, "Hello world.")])
        result = await service.analyze(doc)
        self.assertTrue(result.used_fallback)
        self.assertIn("Source: Uploaded document", result.text)

    async def test_empty_response_uses_fallback(self):
        llm = FakeLlm(response="")
        service = DocumentAnalysisService(llm_generate=llm)
        doc = _make_document([_make_page(1, "Hello world.")])
        result = await service.analyze(doc)
        self.assertTrue(result.used_fallback)

    async def test_fallback_text_is_safe_when_no_text(self):
        # Documents that hit the fallback with no extracted text should still
        # produce a useful response.
        doc = _make_document(
            [_make_page(1, "")],
            filename="empty.pdf",
            status=ExtractionStatus.NO_EXTRACTABLE_TEXT,
        )
        text = _build_fallback_text(doc, truncated=False)
        self.assertIn("Source: Uploaded document", text)
        self.assertIn("empty.pdf", text)


class TestNoLeakage(unittest.IsolatedAsyncioTestCase):
    async def test_document_text_does_not_appear_in_logs(self):
        secret = "TOPSECRET_FINANCIAL_LINE_9999"
        llm = FakeLlm(response="ok")
        service = DocumentAnalysisService(llm_generate=llm)

        class _CaptureHandler(logging.Handler):
            def __init__(self):
                super().__init__()
                self.records = []

            def emit(self, record):
                self.records.append(self.format(record))

        cap = _CaptureHandler()
        cap.setLevel(logging.DEBUG)
        target_logger = logging.getLogger("app.services.documents.document_analysis_service")
        target_logger.addHandler(cap)
        try:
            doc = _make_document(
                [_make_page(1, secret), _make_page(2, "Public line.")],
                filename="secret.pdf",
            )
            with self.assertNoLogs(
                "app.services.documents.document_analysis_service",
                level="DEBUG",
            ):
                await service.analyze(doc)
        finally:
            target_logger.removeHandler(cap)
        # And the LLM prompt should still include it (the LLM is the
        # intended recipient); we only protect logs here.
        self.assertIn(secret, llm.calls[0])
