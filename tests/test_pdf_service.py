"""Local, generated-PDF tests for Phase 3 Step 1 extraction."""

import tempfile
import unittest
from pathlib import Path

import fitz

from app.services.documents.document_models import ExtractionStatus
from app.services.documents.pdf_service import (
    InvalidPdfError,
    PdfExtractionService,
    PdfFileNotFoundError,
)


class TestPdfExtractionService(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.service = PdfExtractionService()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_pdf(self, name, pages, title=None):
        path = self.directory / name
        document = fitz.open()
        if title:
            metadata = document.metadata
            metadata["title"] = title
            document.set_metadata(metadata)
        for text in pages:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(path)
        document.close()
        return path

    def test_valid_multi_page_extraction_preserves_boundaries_and_metadata(self):
        path = self._create_pdf("report.pdf", ["Revenue was $100 million.", "Net income was $20 million."], "Annual Report")
        result = self.service.extract(path)
        self.assertEqual(result.status, ExtractionStatus.SUCCESS)
        self.assertEqual(result.filename, "report.pdf")
        self.assertEqual(result.page_count, 2)
        self.assertEqual([page.page_number for page in result.pages], [1, 2])
        self.assertIn("Revenue was $100 million.", result.pages[0].text)
        self.assertIn("Net income was $20 million.", result.pages[1].text)
        self.assertIn("\n\n", result.extracted_text)
        self.assertEqual(result.title, "Annual Report")

    def test_empty_page_is_retained(self):
        result = self.service.extract(self._create_pdf("empty-page.pdf", ["Cash flow", "", "Debt"]))
        self.assertEqual(result.page_count, 3)
        self.assertEqual(result.pages[1].page_number, 2)
        self.assertEqual(result.pages[1].text, "")
        self.assertEqual(result.pages[1].character_count, 0)

    def test_empty_document_returns_structured_status(self):
        # PyMuPDF refuses to save a zero-page document, so synthesize a minimal
        # raw PDF byte stream with the catalog pointing to an empty page tree.
        minimal_empty_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
            b"xref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000054 00000 n \n"
            b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n98\n%%EOF\n"
        )
        path = self.directory / "empty.pdf"
        path.write_bytes(minimal_empty_pdf)
        result = self.service.extract(path)
        self.assertEqual(result.status, ExtractionStatus.EMPTY_DOCUMENT)
        self.assertEqual(result.page_count, 0)

    def test_image_only_pdf_reports_no_extractable_text(self):
        result = self.service.extract(self._create_pdf("scan.pdf", [""]))
        self.assertEqual(result.status, ExtractionStatus.NO_EXTRACTABLE_TEXT)
        self.assertTrue(result.warnings)

    def test_missing_non_pdf_and_corrupted_files_raise_meaningful_errors(self):
        with self.assertRaises(PdfFileNotFoundError):
            self.service.extract(self.directory / "missing.pdf")

        text_file = self.directory / "notes.txt"
        text_file.write_text("not a PDF", encoding="utf-8")
        with self.assertRaises(InvalidPdfError):
            self.service.extract(text_file)

        corrupt = self.directory / "corrupt.pdf"
        corrupt.write_bytes(b"not a valid PDF")
        with self.assertRaises(InvalidPdfError):
            self.service.extract(corrupt)

    def test_large_page_text_is_preserved(self):
        large_text = "Financial metric 12345. " * 1200
        path = self.directory / "large.pdf"
        document = fitz.open()
        page = document.new_page(width=595, height=10000)
        page.insert_textbox(fitz.Rect(36, 36, 559, 9960), large_text, fontsize=8)
        document.save(path)
        document.close()
        result = self.service.extract(path)
        self.assertGreater(result.pages[0].character_count, 10_000)
        self.assertIn("Financial metric 12345.", result.extracted_text)

    def test_document_text_is_not_logged(self):
        secret = "CONFIDENTIAL_FINANCIAL_VALUE_987654"
        path = self._create_pdf("private.pdf", [secret])
        with self.assertNoLogs("app.services.documents.pdf_service"):
            result = self.service.extract(path)
        self.assertIn(secret, result.extracted_text)


if __name__ == "__main__":
    unittest.main()
