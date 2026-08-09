"""Safe, local, page-preserving PDF text extraction using PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz

from app.services.documents.document_models import (
    ExtractedDocument,
    ExtractedPage,
    ExtractionStatus,
)


class PdfExtractionError(Exception):
    """Base exception for PDF ingestion failures."""


class PdfFileNotFoundError(PdfExtractionError):
    """The requested PDF path does not exist or is not a regular file."""


class InvalidPdfError(PdfExtractionError):
    """The supplied file is not a readable PDF."""


class PasswordProtectedPdfError(PdfExtractionError):
    """The PDF is encrypted and cannot be extracted without a password."""


class PdfExtractionService:
    """Read untrusted PDFs locally without persistence, links, OCR, or network access."""

    def extract(self, file_path: str | Path) -> ExtractedDocument:
        """Extract page text and metadata from a PDF path.

        Image-only/scanned PDFs return ``NO_EXTRACTABLE_TEXT`` rather than an
        apparently successful empty document. A single broken page is retained
        with an error marker while extraction continues for the remaining pages.
        """
        path = Path(file_path)
        if not path.is_file():
            raise PdfFileNotFoundError("PDF file was not found.")
        if path.suffix.lower() != ".pdf":
            raise InvalidPdfError("Only PDF files are supported.")

        try:
            document = fitz.open(path)
        except (fitz.FileDataError, fitz.EmptyFileError, RuntimeError, OSError) as exc:
            raise InvalidPdfError("The file is not a valid readable PDF.") from exc

        try:
            if document.needs_pass:
                raise PasswordProtectedPdfError("Password-protected PDFs are not supported.")

            page_count = document.page_count
            title = _clean_metadata_value(document.metadata.get("title"))
            if page_count == 0:
                return ExtractedDocument(
                    filename=path.name,
                    page_count=0,
                    extracted_text="",
                    pages=[],
                    status=ExtractionStatus.EMPTY_DOCUMENT,
                    title=title,
                    warnings=["The PDF contains no pages."],
                )

            pages: list[ExtractedPage] = []
            warnings: list[str] = []
            for page_index in range(page_count):
                page_number = page_index + 1
                try:
                    text = _clean_page_text(document.load_page(page_index).get_text("text"))
                    pages.append(ExtractedPage(page_number, text, len(text)))
                except Exception:
                    # Do not expose PDF content or machine paths in the result/logs.
                    pages.append(ExtractedPage(page_number, "", 0, "Page text could not be extracted."))
                    warnings.append(f"Text could not be extracted from page {page_number}.")

            extracted_text = "\n\n".join(page.text for page in pages if page.text)
            if not extracted_text:
                status = ExtractionStatus.NO_EXTRACTABLE_TEXT
                warnings.append("No extractable text was found; the PDF may be scanned or image-only.")
            elif any(page.extraction_error for page in pages):
                status = ExtractionStatus.PARTIAL
            else:
                status = ExtractionStatus.SUCCESS

            return ExtractedDocument(
                filename=path.name,
                page_count=page_count,
                extracted_text=extracted_text,
                pages=pages,
                status=status,
                title=title,
                warnings=warnings,
            )
        finally:
            document.close()


def _clean_page_text(text: str) -> str:
    """Remove only safe extraction artifacts while preserving content and numbers."""
    return (text or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _clean_metadata_value(value) -> str | None:
    value = (value or "").strip()
    return value or None


pdf_extraction_service = PdfExtractionService()
