"""Tests for Snowflake staged OCR normalization and cleanup."""

from types import SimpleNamespace

import pytest

from knowledge_agent.claims.ocr import SnowflakeParseDocumentOcrClient


class FakeDataFrame:
    def __init__(self, rows=None, error: Exception | None = None):
        self.rows = rows or []
        self.error = error

    def collect(self):
        if self.error is not None:
            raise self.error
        return self.rows


class FakeFileOperations:
    def __init__(self):
        self.put_calls = []
        self.remove_calls = []
        self.put_results = [SimpleNamespace(status="UPLOADED", message="")]
        self.remove_error = None

    def put(self, local_file_name, stage_location, **kwargs):
        self.put_calls.append((local_file_name, stage_location, kwargs))
        return self.put_results

    def remove(self, stage_location):
        self.remove_calls.append(stage_location)
        if self.remove_error is not None:
            raise self.remove_error
        return []


class FakeSession:
    def __init__(self, parsed_result=None):
        self.file = FakeFileOperations()
        self.sql_calls = []
        self.parsed_result = parsed_result

    def get_current_database(self):
        return "KNOWLEDGE_AGENT_DB"

    def get_current_schema(self):
        return "PUBLIC"

    def sql(self, query, params=None):
        self.sql_calls.append((query, params))
        if query.lstrip().startswith("SELECT"):
            return FakeDataFrame([{"PARSED_DOCUMENT": self.parsed_result}])
        return FakeDataFrame()


def valid_result():
    return {
        "error": None,
        "value": {
            "pages": [
                {"index": 1, "content": " Second page "},
                {"index": 0, "content": "First page"},
            ]
        },
        "metadata": {"pageCount": 2},
    }


def test_snowflake_ocr_creates_stage_parses_pages_and_removes_upload(
    monkeypatch, tmp_path
):
    session = FakeSession(valid_result())
    monkeypatch.setattr(
        "knowledge_agent.claims.ocr.uuid4",
        lambda: SimpleNamespace(hex="run-id"),
    )
    pdf_path = tmp_path / "claim.pdf"
    pdf_path.write_bytes(b"pdf")

    client = SnowflakeParseDocumentOcrClient(
        session,
        "KNOWLEDGE_AGENT_DOCUMENTS",
    )
    pages = client.extract_pages("CLM-001", pdf_path)

    assert [(page.page_number, page.page_id, page.text) for page in pages] == [
        (1, "CLM-001:p1", "First page"),
        (2, "CLM-001:p2", "Second page"),
    ]
    assert session.sql_calls[0] == (
        "CREATE STAGE IF NOT EXISTS IDENTIFIER(?)",
        ["KNOWLEDGE_AGENT_DOCUMENTS"],
    )
    assert session.sql_calls[1][1] == [
        "@KNOWLEDGE_AGENT_DOCUMENTS",
        "knowledge-agent/run-id/claim.pdf",
    ]
    assert session.file.put_calls == [
        (
            str(pdf_path),
            "@KNOWLEDGE_AGENT_DOCUMENTS/knowledge-agent/run-id",
            {"auto_compress": False, "overwrite": False},
        )
    ]
    assert session.file.remove_calls == [
        "@KNOWLEDGE_AGENT_DOCUMENTS/knowledge-agent/run-id"
    ]


def test_snowflake_ocr_surfaces_provider_error_and_still_removes_upload(tmp_path):
    session = FakeSession(
        {
            "error": "document could not be parsed",
            "value": None,
            "metadata": {},
        }
    )
    pdf_path = tmp_path / "claim.pdf"
    pdf_path.write_bytes(b"pdf")
    client = SnowflakeParseDocumentOcrClient(session, "OCR_STAGE")

    with pytest.raises(RuntimeError, match="document could not be parsed"):
        client.extract_pages("CLM-001", pdf_path)

    assert len(session.file.remove_calls) == 1


def test_snowflake_ocr_rejects_failed_upload_and_cleans_prefix(tmp_path):
    session = FakeSession(valid_result())
    session.file.put_results = [
        SimpleNamespace(status="FAILED", message="upload denied")
    ]
    pdf_path = tmp_path / "claim.pdf"
    pdf_path.write_bytes(b"pdf")
    client = SnowflakeParseDocumentOcrClient(session, "OCR_STAGE")

    with pytest.raises(RuntimeError, match="FAILED.*upload denied"):
        client.extract_pages("CLM-001", pdf_path)

    assert len(session.file.remove_calls) == 1


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({"error": None, "value": {}, "metadata": {}}, "non-empty pages"),
        (
            {
                "error": None,
                "value": {"pages": [{"index": 1, "content": "page"}]},
                "metadata": {"pageCount": 1},
            },
            "contiguous from zero",
        ),
        (
            {
                "error": None,
                "value": {"pages": [{"index": 0, "content": "page"}]},
                "metadata": {"pageCount": 2},
            },
            "page count does not match",
        ),
        (
            {
                "error": None,
                "value": {"pages": [{"index": 0, "content": 4}]},
                "metadata": {"pageCount": 1},
            },
            "content must be text",
        ),
    ],
)
def test_snowflake_ocr_rejects_malformed_results(tmp_path, result, message):
    session = FakeSession(result)
    pdf_path = tmp_path / "claim.pdf"
    pdf_path.write_bytes(b"pdf")
    client = SnowflakeParseDocumentOcrClient(session, "OCR_STAGE")

    with pytest.raises(ValueError, match=message):
        client.extract_pages("CLM-001", pdf_path)

    assert len(session.file.remove_calls) == 1


def test_snowflake_ocr_rejects_missing_database_context():
    session = FakeSession(valid_result())
    session.get_current_database = lambda: None

    with pytest.raises(ValueError, match="database and schema"):
        SnowflakeParseDocumentOcrClient(session, "OCR_STAGE")


def test_snowflake_ocr_surfaces_stage_creation_and_cleanup_failures(tmp_path):
    class StageFailureSession(FakeSession):
        def sql(self, query, params=None):
            return FakeDataFrame(error=RuntimeError("cannot create stage"))

    with pytest.raises(RuntimeError, match="cannot create stage"):
        SnowflakeParseDocumentOcrClient(StageFailureSession(), "OCR_STAGE")

    session = FakeSession(valid_result())
    session.file.remove_error = RuntimeError("cannot remove staged evidence")
    pdf_path = tmp_path / "claim.pdf"
    pdf_path.write_bytes(b"pdf")
    client = SnowflakeParseDocumentOcrClient(session, "OCR_STAGE")

    with pytest.raises(RuntimeError, match="cannot remove staged evidence"):
        client.extract_pages("CLM-001", pdf_path)
