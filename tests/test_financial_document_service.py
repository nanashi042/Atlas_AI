"""Unit tests for bounded financial document intelligence."""

import unittest

from app.services.documents.document_models import ExtractedDocument, ExtractedPage, ExtractionStatus
from app.services.documents.document_qa_service import DocumentContextStore
from app.services.documents.financial_document_service import (
    FINANCIAL_UNAVAILABLE_MESSAGE,
    INSUFFICIENT_FINANCIAL_INFORMATION_MESSAGE,
    FinancialDocumentService,
    compare_metric,
    extract_financial_insights,
)


def context(*pages):
    document = ExtractedDocument(
        filename="annual.pdf", page_count=len(pages), extracted_text="\n\n".join(text for text in pages),
        pages=[ExtractedPage(index, text, len(text)) for index, text in enumerate(pages, 1)],
        status=ExtractionStatus.SUCCESS, title="Annual Report",
    )
    return DocumentContextStore().set_document("user", document)


class FakeLlm:
    def __init__(self, response="Revenue increased.\n\nSource: Page 1", error=None):
        self.response, self.error, self.prompts = response, error, []

    async def __call__(self, prompt, history=None):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.response


class TestMetricExtraction(unittest.TestCase):
    def test_metrics_and_page_references_are_extracted(self):
        insights = extract_financial_insights(context(
            "Revenue was $12.4B in 2025. Revenue was $10.1B in 2024. Net income was $2.2B in 2025."
        ).document)
        revenue = insights.metrics["revenue"]
        self.assertEqual({item.fiscal_year for item in revenue.values}, {2024, 2025})
        self.assertTrue(all(item.page_number == 1 for item in revenue.values))
        self.assertIn("net_income", insights.metrics)

    def test_revenue_comparison_calculates_percentage(self):
        insights = extract_financial_insights(context("Revenue was $12.0B in 2025. Revenue was $10.0B in 2024.").document)
        new, old, growth = compare_metric(insights.metrics["revenue"])
        self.assertEqual((new.fiscal_year, old.fiscal_year), (2025, 2024))
        self.assertAlmostEqual(growth, 20.0)

    def test_tabular_style_year_value_pairs_are_not_mistaken_for_years(self):
        insights = extract_financial_insights(context("Revenue: 2025: $12B; 2024: $10B.").document)
        values = {item.fiscal_year: item.value for item in insights.metrics["revenue"].values}
        self.assertEqual(values, {2025: "$12B", 2024: "$10B"})

    def test_net_income_comparison_and_missing_metric(self):
        insights = extract_financial_insights(context("Net income was $3M in 2025. Net income was $2M in 2024.").document)
        self.assertAlmostEqual(compare_metric(insights.metrics["net_income"])[2], 50.0)
        self.assertNotIn("revenue", insights.metrics)

    def test_risks_are_kept_with_their_document_page(self):
        insights = extract_financial_insights(context("Competition may adversely affect margins and is a significant risk.").document)
        self.assertEqual(insights.risks[0][1], 1)
        self.assertIn("risk", insights.risks[0][0].lower())


class TestFinancialDocumentService(unittest.IsolatedAsyncioTestCase):
    async def test_financial_performance_risk_and_bull_bear_prompt_is_grounded(self):
        active = context("Revenue was $12B in 2025. Revenue was $10B in 2024. Competition is a risk.")
        llm = FakeLlm("📈 Positive\nRevenue grew.\n\n📉 Concerns\nCompetition.\n\nSources: Pages 1")
        result = await FinancialDocumentService(llm_generate=llm).answer("Give me bullish and bearish points.", active)
        self.assertIn("📊 Financial Insight", result)
        self.assertIn("not financial advice", result)
        self.assertIn("STRUCTURED FINANCIAL METRICS", llm.prompts[0])
        self.assertIn("Answer ONLY", llm.prompts[0])
        self.assertIn("[Page 1]", llm.prompts[0])

    async def test_unsupported_financial_question_returns_safe_message(self):
        active = context("This letter contains no quantitative financial information.")
        service = FinancialDocumentService(llm_generate=FakeLlm())
        self.assertEqual(await service.answer("What was revenue growth?", active), INSUFFICIENT_FINANCIAL_INFORMATION_MESSAGE)

    async def test_gemini_failure_does_not_log_document_content(self):
        secret = "CONFIDENTIAL_FINANCIAL_VALUE_777"
        active = context(f"Revenue was {secret} in 2025.")
        service = FinancialDocumentService(llm_generate=FakeLlm(error=RuntimeError("offline")))
        with self.assertLogs("app.services.documents.financial_document_service", level="ERROR") as logs:
            result = await service.answer("What was revenue?", active)
        self.assertEqual(result, FINANCIAL_UNAVAILABLE_MESSAGE)
        self.assertNotIn(secret, "\n".join(logs.output))

    async def test_bounded_context_is_disclosed_to_model_and_user(self):
        active = context("Revenue was $12B in 2025.")
        active = type(active)(document=active.document, truncated=True)
        llm = FakeLlm()
        result = await FinancialDocumentService(llm_generate=llm).answer("What was revenue?", active)
        self.assertIn("bounded extract", llm.prompts[0])
        self.assertIn("available bounded extract", result)
