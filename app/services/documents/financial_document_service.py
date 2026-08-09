"""Bounded, document-only financial intelligence for uploaded PDFs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.ai.llm import generate_response
from app.services.documents.document_models import ExtractedDocument
from app.services.documents.document_qa_service import ActiveDocumentContext, _render_pages

logger = logging.getLogger(__name__)

FINANCIAL_UNAVAILABLE_MESSAGE = (
    "⚠️ I couldn't generate a financial insight from the uploaded document right now. "
    "Please try again shortly."
)
INSUFFICIENT_FINANCIAL_INFORMATION_MESSAGE = (
    "I couldn't find enough supported financial information in the uploaded document to answer that."
)


@dataclass(frozen=True)
class FinancialMetricValue:
    value: str
    fiscal_year: Optional[int]
    page_number: int
    numeric_value: Optional[float] = None


@dataclass
class FinancialMetric:
    name: str
    values: list[FinancialMetricValue] = field(default_factory=list)


@dataclass
class FinancialDocumentInsights:
    metrics: dict[str, FinancialMetric] = field(default_factory=dict)
    risks: list[tuple[str, int]] = field(default_factory=list)

    def pages(self) -> list[int]:
        return sorted({value.page_number for metric in self.metrics.values() for value in metric.values} | {page for _, page in self.risks})


_METRIC_ALIASES = {
    "revenue": ("revenue", "net sales", "sales"),
    "operating_income": ("operating income", "income from operations"),
    "net_income": ("net income", "net earnings"),
    "operating_margin": ("operating margin",),
    "gross_margin": ("gross margin",),
    "eps": ("earnings per share", "eps"),
    "cash": ("cash and cash equivalents", "cash"),
    "debt": ("total debt", "long-term debt", "debt"),
    "free_cash_flow": ("free cash flow",),
    "capex": ("capital expenditures", "capex"),
    "research_and_development": ("research and development", "r&d"),
    "employees": ("employees", "headcount"),
}
_VALUE = r"(?:[$€£]\s?)?\d[\d,]*(?:\.\d+)?\s?(?:billion|million|thousand|[BMK])?%?"


def _numeric_value(value: str) -> Optional[float]:
    """Normalize a displayed financial number solely for comparison math."""
    match = re.search(r"\d[\d,]*(?:\.\d+)?", value)
    if not match:
        return None
    amount = float(match.group(0).replace(",", ""))
    suffix = value.lower()
    if "billion" in suffix or re.search(r"\b\d[\d,.]*\s*b\b", suffix):
        amount *= 1_000_000_000
    elif "million" in suffix or re.search(r"\b\d[\d,.]*\s*m\b", suffix):
        amount *= 1_000_000
    elif "thousand" in suffix or re.search(r"\b\d[\d,.]*\s*k\b", suffix):
        amount *= 1_000
    return amount


def _add_value(insights: FinancialDocumentInsights, name: str, value: str, year: Optional[int], page: int) -> None:
    metric = insights.metrics.setdefault(name, FinancialMetric(name))
    candidate = FinancialMetricValue(value=value.strip(), fiscal_year=year, page_number=page, numeric_value=_numeric_value(value))
    if candidate not in metric.values:
        metric.values.append(candidate)


def extract_financial_insights(document: ExtractedDocument) -> FinancialDocumentInsights:
    """Extract only explicit metric/year/value pairs and risk statements.

    This intentionally modest parser supplements Gemini; it does not infer a
    value from tables or prose that cannot be safely matched.
    """
    insights = FinancialDocumentInsights()
    for page in document.pages:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", page.text or ""):
            lower = sentence.lower()
            for name, aliases in _METRIC_ALIASES.items():
                alias_match = next((re.search(rf"\b{re.escape(alias)}\b", lower) for alias in aliases if re.search(rf"\b{re.escape(alias)}\b", lower)), None)
                if not alias_match:
                    continue
                # Support both "Revenue was $12B in 2025" and tabular-style
                # "Revenue: 2025: $12B; 2024: $10B" lines. Prefer year/value
                # pairs when present so a year is never mistaken for a value.
                nearby = sentence[alias_match.end():]
                year_value_pairs = list(re.finditer(rf"(20\d{{2}}).{{0,12}}?({_VALUE})", nearby, re.IGNORECASE))
                if year_value_pairs:
                    for match in year_value_pairs:
                        _add_value(insights, name, match.group(2), int(match.group(1)), page.page_number)
                else:
                    for match in re.finditer(rf"({_VALUE}).{{0,20}}?(20\d{{2}})", nearby, re.IGNORECASE):
                        value = match.group(1)
                        # A bare four-digit year is not a financial amount.
                        if re.fullmatch(r"20\d{2}", value.strip()):
                            continue
                        _add_value(insights, name, value, int(match.group(2)), page.page_number)
                # A value with no year remains useful but cannot support a period comparison.
                if not insights.metrics.get(name, FinancialMetric(name)).values:
                    value_match = re.search(rf"(?:{'|'.join(re.escape(alias) for alias in aliases)}).{{0,40}}?({_VALUE})", sentence, re.IGNORECASE)
                    if value_match:
                        _add_value(insights, name, value_match.group(1), None, page.page_number)
            if any(term in lower for term in ("risk", "risks", "may adversely", "could adversely", "competition", "uncertainty")):
                cleaned = " ".join(sentence.split())
                if len(cleaned) >= 20:
                    item = (cleaned, page.page_number)
                    if item not in insights.risks:
                        insights.risks.append(item)
    return insights


def compare_metric(metric: FinancialMetric) -> Optional[tuple[FinancialMetricValue, FinancialMetricValue, Optional[float]]]:
    """Return newest, oldest, and calculated percentage change when possible."""
    dated = sorted((value for value in metric.values if value.fiscal_year is not None), key=lambda item: item.fiscal_year)
    if len(dated) < 2:
        return None
    old, new = dated[0], dated[-1]
    growth = None
    if old.numeric_value not in (None, 0) and new.numeric_value is not None:
        growth = ((new.numeric_value - old.numeric_value) / old.numeric_value) * 100
    return new, old, growth


def _render_metrics(insights: FinancialDocumentInsights) -> str:
    if not insights.metrics:
        return "No explicit structured financial metrics were safely extracted."
    lines = []
    for name, metric in insights.metrics.items():
        values = "; ".join(f"{value.fiscal_year or 'period not stated'}: {value.value} (Page {value.page_number})" for value in metric.values)
        lines.append(f"- {name}: {values}")
    return "\n".join(lines)


_FINANCIAL_PROMPT = """You are Atlas AI's financial document intelligence assistant.
Answer ONLY from the supplied bounded uploaded-document context and structured extraction. Do not use outside knowledge or invent facts, metrics, risks, dates, comparisons, or page numbers.
For risks, label direct document statements as "Stated risk" and any careful inference as "Interpretation". For bullish/bearish points, ground every point in the document and say informational analysis only, not financial advice. If information is insufficient, say so clearly. Preserve currency, units, and dates exactly. Keep the response concise and mobile friendly. Include Source: Page N or Sources: Pages N, M only for supplied page markers. Do not add a heading.

QUESTION:
{question}

STRUCTURED FINANCIAL METRICS:
{metrics}

{truncation_notice}
DOCUMENT CONTEXT:
{pages}
"""


class FinancialDocumentService:
    """Generates grounded financial comparisons, risks, and performance insight."""

    def __init__(self, llm_generate=None):
        self.llm_generate = llm_generate

    async def answer(self, question: str, context: ActiveDocumentContext) -> str:
        insights = extract_financial_insights(context.document)
        if not insights.metrics and not insights.risks:
            return INSUFFICIENT_FINANCIAL_INFORMATION_MESSAGE
        prompt = _FINANCIAL_PROMPT.format(
            question=question,
            metrics=_render_metrics(insights),
            truncation_notice=("Important: this is a bounded extract; do not claim unavailable pages were checked." if context.truncated else ""),
            pages=_render_pages(context.document),
        )
        try:
            response = await self._call_llm(prompt)
        except Exception as exc:
            logger.error("Financial document Gemini call failed: %s", exc)
            return FINANCIAL_UNAVAILABLE_MESSAGE
        response = (response or "").strip()
        if not response or response.startswith("[Error") or response.startswith("⚠️"):
            logger.warning("Financial document service received no usable Gemini response.")
            return FINANCIAL_UNAVAILABLE_MESSAGE
        text = f"📊 Financial Insight\n\n{response}\n\nInformational analysis only — not financial advice."
        if context.truncated:
            text += "\n\n⚠️ Based on the available bounded extract."
        return text

    async def _call_llm(self, prompt: str) -> str:
        if self.llm_generate is not None:
            return await self.llm_generate(prompt, history=[])
        return await generate_response(prompt, history=[])


def is_financial_document_question(message: str) -> bool:
    text = (message or "").lower()
    phrases = (
        "revenue", "net income", "operating income", "operating margin", "gross margin", "eps", "cash flow", "free cash flow", "capex", "r&d", "research and development", "financial performance", "financial metrics", "bullish", "bearish", "risk", "risks", "risk factors", "compare 20", "what grew", "which page mentions",
    )
    return any(phrase in text for phrase in phrases)


financial_document_service = FinancialDocumentService()
