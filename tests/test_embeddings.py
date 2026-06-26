from claim_kb.config import ClaimKbSettings
from claim_kb.embeddings import SnowflakeAiEmbedder


class FakeSession:
    def __init__(self):
        self.closed = False
        self.rows = []

    def create_dataframe(self, rows, schema):
        self.rows = rows
        return FakeDataFrame(rows)

    def close(self):
        self.closed = True


class FakeDataFrame:
    def __init__(self, rows):
        self.rows = list(rows)
        self.sorted = False

    def select(self, *columns):
        return self

    def sort(self, *columns):
        self.sorted = True
        return self

    def collect(self):
        rows = sorted(self.rows) if self.sorted else list(reversed(self.rows))
        return [
            {"CHUNK_INDEX": index, "EMBEDDING": [float(index), float(len(text))]}
            for index, text in rows
        ]


def test_snowflake_ai_embedder_returns_ordered_embeddings():
    session = FakeSession()
    settings = ClaimKbSettings(
        data_root="data/claims",
        ai_project_endpoint="https://example.services.ai.azure.com/api/projects/proj",
        document_intelligence_endpoint="https://example.cognitiveservices.azure.com",
        chat_deployment="gpt-test",
        tenant_id=None,
        snowflake_connection_name="default",
        snowflake_embedding_model="snowflake-arctic-embed-l-v2.0",
    )
    embedder = SnowflakeAiEmbedder(settings, session=session)

    embeddings = embedder.embed_texts(["alpha", "beta"])
    embedder.close()

    assert embeddings == [[0.0, 5.0], [1.0, 4.0]]
    assert embedder.embedding_provider == "snowflake"
    assert embedder.embedding_model == "snowflake-arctic-embed-l-v2.0"
    assert not session.closed
