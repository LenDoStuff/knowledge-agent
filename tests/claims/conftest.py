from pathlib import Path

import pytest
from pypdf import PdfWriter


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "scanned_claim.pdf"
    writer = PdfWriter()
    for _ in range(4):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)
    return path
