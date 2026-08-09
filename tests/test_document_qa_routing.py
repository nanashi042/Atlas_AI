"""Regression tests for document-Q&A manager routing precedence."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.manager import process_message
from app.services.documents.document_models import ExtractedDocument, ExtractedPage, ExtractionStatus
from app.services.documents.document_qa_service import (
    DOCUMENT_CONTEXT_CLEARED_MESSAGE,
    NO_ACTIVE_DOCUMENT_MESSAGE,
    document_context_store,
)


def active_document(user_id="document-routing-user"):
    document_context_store.clear_document(user_id)
    document_context_store.set_document(user_id, ExtractedDocument(
        filename="report.pdf", page_count=1, extracted_text="Revenue was $12.4 billion.",
        pages=[ExtractedPage(1, "Revenue was $12.4 billion.", 26)],
        status=ExtractionStatus.SUCCESS, title="Report",
    ))
    return user_id


class TestDocumentQaRouting(unittest.IsolatedAsyncioTestCase):
    async def test_active_document_question_routes_to_qa(self):
        user_id = active_document()
        with patch("app.agent.manager.financial_document_service.answer", new=AsyncMock(return_value="financial answer")) as financial, \
             patch("app.agent.manager.document_qa_service.answer", new=AsyncMock(return_value="document answer")) as answer:
            result = await process_message("What was the revenue?", user_id)
        self.assertEqual(result, "financial answer")
        financial.assert_awaited_once()
        answer.assert_not_awaited()

    async def test_financial_question_routes_to_financial_service_and_general_qa_remains_available(self):
        user_id = active_document()
        with patch("app.agent.manager.financial_document_service.answer", new=AsyncMock(return_value="financial")) as financial, \
             patch("app.agent.manager.document_qa_service.answer", new=AsyncMock(return_value="general")) as general:
            self.assertEqual(await process_message("What was the revenue?", user_id), "financial")
            self.assertEqual(await process_message("What does this section mean?", user_id), "general")
        financial.assert_awaited_once()
        general.assert_awaited_once()

    async def test_explicit_question_without_document_requests_upload(self):
        user_id = "no-document-user"
        document_context_store.clear_document(user_id)
        self.assertEqual(
            await process_message("According to the PDF, what was revenue?", user_id),
            NO_ACTIVE_DOCUMENT_MESSAGE,
        )

    async def test_natural_and_command_style_clear_remove_context(self):
        user_id = active_document()
        self.assertEqual(await process_message("Forget this document", user_id), DOCUMENT_CONTEXT_CLEARED_MESSAGE)
        self.assertIsNone(document_context_store.get_document(user_id))

    async def test_existing_company_research_wins_over_document_qa(self):
        user_id = active_document()
        with patch("app.agent.manager.document_qa_service.answer", new=AsyncMock()) as qa, \
             patch("app.agent.manager.CompanyResearchService.get_company_research", new=AsyncMock(return_value=MagicMock())), \
             patch("app.agent.manager.research_company_ai", new=AsyncMock(return_value="research")):
            result = await process_message("What is NVDA?", user_id)
        self.assertEqual(result, "research")
        qa.assert_not_awaited()

    async def test_watchlist_alert_and_briefing_win_over_document_qa(self):
        user_id = active_document()
        with patch("app.agent.manager.document_qa_service.answer", new=AsyncMock()) as qa, \
             patch("app.agent.manager._handle_watchlist_list", new=AsyncMock(return_value="watchlist")), \
             patch("app.agent.manager._handle_alert_list", new=AsyncMock(return_value="alerts")), \
             patch("app.agent.manager.briefing_service.generate_briefing", new=AsyncMock(return_value="briefing")):
            self.assertEqual(await process_message("Show my watchlist", user_id), "watchlist")
            self.assertEqual(await process_message("Show my alerts", user_id), "alerts")
            self.assertEqual(await process_message("Give me today's briefing", user_id), "briefing")
        qa.assert_not_awaited()
