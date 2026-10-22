import os
import sys
import unittest
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.compile_paper import compile_manuscript


class TestManuscriptCompilation(unittest.TestCase):
    def setUp(self):
        self.paper_dir = PROJECT_ROOT / "paper"
        self.review_tex = self.paper_dir / "manuscript_review.tex"
        self.submission_tex = self.paper_dir / "manuscript_submission.tex"
        self.bib_path = self.paper_dir / "references.bib"
        self.audit_doc = PROJECT_ROOT / "docs" / "manuscript_evidence_reconciliation_audit.md"

    def test_manuscript_review_contains_all_notes(self):
        """C13: manuscript_review.tex preserves all 56 review notes ([Nota v2: ...])."""
        self.assertTrue(self.review_tex.exists(), "manuscript_review.tex missing")
        content = self.review_tex.read_text(encoding="utf-8")
        # Check presence of review macro or notes
        import re
        notes = re.findall(r"\\notavdois\{|\[Nota v2:", content)
        self.assertGreaterEqual(len(notes), 56)

    def test_manuscript_submission_clean_of_notes(self):
        """C14: manuscript_submission.tex excises all review notes and macros."""
        self.assertTrue(self.submission_tex.exists(), "manuscript_submission.tex missing")
        content = self.submission_tex.read_text(encoding="utf-8")
        self.assertNotIn("[Nota v2:", content)
        self.assertNotIn(r"\notavdois{", content)

    def test_references_bib_required_citations(self):
        """C15: references.bib contains complete bibliographic entries for required citations."""
        self.assertTrue(self.bib_path.exists(), "references.bib missing")
        bib_content = self.bib_path.read_text(encoding="utf-8").lower()
        self.assertIn("rojas-carulla", bib_content)  # GeNet
        self.assertIn("loshchilov", bib_content)     # AdamW
        self.assertIn("gotmare", bib_content)       # Warmup/fine-tuning
        self.assertIn("mcinnes", bib_content)       # UMAP
        self.assertIn("sharp", bib_content)         # RSCU

    def test_compile_paper_submission_exit_zero(self):
        """C16: compile_manuscript on submission mode exits with code 0."""
        # Test compiling manuscript_submission.tex
        res = compile_manuscript(self.submission_tex, output_dir=self.paper_dir)
        self.assertEqual(res.returncode, 0, f"Compilation failed: {res.stderr}\n{res.stdout}")
        submission_pdf = self.paper_dir / "manuscript_submission.pdf"
        self.assertTrue(submission_pdf.exists())

    def test_traceability_matrix_covers_all_tables_and_figures(self):
        """C17: audit doc contains mapping for all 12 tables and 5 figures."""
        self.assertTrue(self.audit_doc.exists(), "audit doc missing")
        content = self.audit_doc.read_text(encoding="utf-8")
        for i in range(1, 13):
            self.assertIn(f"Table {i}", content)
        for i in range(1, 6):
            self.assertIn(f"Figure {i}", content)

    def test_submission_title_moderated(self):
        """C18: submission manuscript title matches exact moderated string."""
        content = self.submission_tex.read_text(encoding="utf-8")
        expected_title = "Base-level one-hot 1D convolutional networks for high-cardinality viral taxonomic classification: accuracy, corruption trade-offs, and learned sequence representations"
        self.assertIn(expected_title, content)
        audit_content = self.audit_doc.read_text(encoding="utf-8")
        self.assertIn(expected_title, audit_content)

    def test_abstract_bmc_structured_sections(self):
        """C19: submission abstract contains explicit structured BMC headings."""
        content = self.submission_tex.read_text(encoding="utf-8")
        abstract_start = content.find(r"\begin{abstract}")
        abstract_end = content.find(r"\end{abstract}")
        self.assertNotEqual(abstract_start, -1)
        self.assertNotEqual(abstract_end, -1)
        abstract_text = content[abstract_start:abstract_end]
        self.assertIn(r"\textbf{Background}", abstract_text)
        self.assertIn(r"\textbf{Methods}", abstract_text)
        self.assertIn(r"\textbf{Results}", abstract_text)
        self.assertIn(r"\textbf{Conclusions}", abstract_text)

    def test_scientific_evidence_demarcations(self):
        """C20: submission manuscript documents scientific demarcations and caveats."""
        content = self.submission_tex.read_text(encoding="utf-8")
        # GeNet
        self.assertIn("GeNet", content)
        # 80/10/10 fragment split vs genome disjoint
        self.assertIn("80/10/10", content)
        # seen contig reference overlap / long sequence explanation
        self.assertIn("Kraken", content)
        self.assertIn("72.69", content)
        # Availability
        self.assertIn("Availability of data and materials", content)


if __name__ == "__main__":
    unittest.main()
