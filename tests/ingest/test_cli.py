import pytest

from ingest.cli import build_parser


def test_cli_accepts_exactly_one_ingestion_input():
    parser = build_parser()

    pdf_args = parser.parse_args(
        ["--claim-id", "CLM-001", "--pdf-path", "claim.pdf"]
    )
    folder_args = parser.parse_args(
        ["--claim-id", "CLM-001", "--folder-path", "documents"]
    )

    assert pdf_args.pdf_path == "claim.pdf"
    assert folder_args.folder_path == "documents"

    with pytest.raises(SystemExit):
        parser.parse_args(["--claim-id", "CLM-001"])

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--claim-id",
                "CLM-001",
                "--pdf-path",
                "claim.pdf",
                "--folder-path",
                "documents",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--claim-id",
                "CLM-001",
                "--pdf-path",
                "claim.pdf",
                "--embedding-mode",
                "none",
            ]
        )
