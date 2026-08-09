"""Structured, in-memory representations of extracted documents."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    EMPTY_DOCUMENT = "empty_document"
    NO_EXTRACTABLE_TEXT = "no_extractable_text"


@dataclass(frozen=True)
class ExtractedPage:
    """Text extracted from one PDF page; page numbers are one-based."""

    page_number: int
    text: str
    character_count: int
    extraction_error: Optional[str] = None


@dataclass(frozen=True)
class ExtractedDocument:
    """A non-persistent PDF extraction result for future document analysis."""

    filename: str
    page_count: int
    extracted_text: str
    pages: list[ExtractedPage]
    status: ExtractionStatus
    title: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
