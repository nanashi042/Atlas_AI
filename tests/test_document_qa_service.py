"""Focused tests for bounded, in-memory document question answering."""

import logging
import unittest

from app.config.settings import settings
from app.services.documents.document_models import ExtractedDocument, ExtractedPage, ExtractionStatus
from app.services.documents.document_qa_service import (
    DOCUMENT_QA_UNAVAILABLE_MESSAGE,
    DocumentContextStore,
    DocumentQaService,
)


def make_document(pages, filename="report.pdf"):
    return ExtractedDocument(
        filename=filename,
        page_count=len(pages),
        extracted_text="\n\n".join(page.text for page in pages if page.text),
        pages=pages,
        status=ExtractionStatus.SUCCESS,
        title="Annual Report",
    )


def page(number, text):
    return ExtractedPage(number, text, len(text))


class FakeLlm:
    def __init__(self, response="Answer\n\nSource: Page 2", error=None):
        self.response = response
        self.error = error
        self.prompts = []

    async def __call__(self, prompt, history=None):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.response


class TestDocumentContextStore(unittest.TestCase):
    def test_new_document_replaces_previous_and_can_be_cleared(self):
        store = DocumentContextStore()
        store.set_document("u1", make_document([page(1, "old revenue")], "old.pdf"))
        store.set_document("u1", make_document([page(1, "new revenue")], "new.pdf"))
        self.assertEqual(store.get_document("u1").document.filename, "new.pdf")
        self.assertTrue(store.clear_document("u1"))
        self.assertIsNone(store.get_document("u1"))

    def test_context_is_bounded_and_preserves_page_markers(self):
        store = DocumentContextStore()
        pages = [page(index, f"page {index}") for index in range(1, settings.DOCUMENT_MAX_PAGES + 3)]
        context = store.set_document("u1", make_document(pages))
        self.assertTrue(context.truncated)
        self.assertEqual(context.document.pages[-1].page_number, settings.DOCUMENT_MAX_PAGES)
        self.assertNotIn(f"page {settings.DOCUMENT_MAX_PAGES + 1}", context.document.extracted_text)


class TestDocumentQaService(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_uses_only_active_document_and_preserves_pages(self):
        store = DocumentContextStore()
        context = store.set_document("u1", make_document([page(1, "Revenue was $10."), page(2, "Risks included debt.")]))
        llm = FakeLlm()
        result = await DocumentQaService(llm_generate=llm).answer("What were the risks?", context)
        self.assertIn("📄 From the uploaded document", result)
        self.assertIn("Source: Page 2", result)
        self.assertIn("[Page 1]", llm.prompts[0])
        self.assertIn("[Page 2]", llm.prompts[0])
        self.assertIn("Use ONLY", llm.prompts[0])

    async def test_unsupported_answer_is_not_fabricated_by_service(self):
        store = DocumentContextStore()
        context = store.set_document("u1", make_document([page(1, "Revenue was $10.")]))
        llm = FakeLlm("I couldn't find that information in the uploaded document.")
        result = await DocumentQaService(llm_generate=llm).answer("What was 2025 revenue?", context)
        self.assertIn("couldn't find that information", result)

    async def test_gemini_failure_is_safe_and_document_text_is_not_logged(self):
        secret = "SECRET_DOCUMENT_VALUE_123"
        store = DocumentContextStore()
        context = store.set_document("u1", make_document([page(1, secret)]))
        service = DocumentQaService(llm_generate=FakeLlm(error=RuntimeError("offline")))
        with self.assertLogs("app.services.documents.document_qa_service", level="ERROR") as logs:
            result = await service.answer("What is the value?", context)
        self.assertEqual(result, DOCUMENT_QA_UNAVAILABLE_MESSAGE)
        self.assertNotIn(secret, "\n".join(logs.output))
