"""Snowflake Cortex embedding adapter."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from claim_kb.config import ClaimKbSettings


class TextEmbedder(Protocol):
    embedding_provider: str
    embedding_model: str

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class SnowflakeAiEmbedder:
    embedding_provider = "snowflake"

    def __init__(
        self,
        settings: ClaimKbSettings,
        session: Any | None = None,
    ) -> None:
        self.embedding_model = settings.snowflake_embedding_model
        self._session = session or create_snowflake_session(settings)
        self._owns_session = session is None

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        from snowflake.snowpark.functions import ai_embed, col

        rows = [(index, text or "") for index, text in enumerate(texts)]
        dataframe = self._session.create_dataframe(rows, schema=["chunk_index", "text"])
        result_rows = (
            dataframe.select(
                col("chunk_index").alias("chunk_index"),
                ai_embed(self.embedding_model, col("text")).alias("embedding"),
            )
            .sort(col("chunk_index"))
            .collect()
        )

        embeddings_by_index: dict[int, list[float]] = {}
        for row in result_rows:
            index = int(_row_get(row, "chunk_index", 0))
            embeddings_by_index[index] = _coerce_embedding(_row_get(row, "embedding", 1))
        return [embeddings_by_index[index] for index in range(len(texts))]

    def close(self) -> None:
        if self._session is not None and self._owns_session:
            close = getattr(self._session, "close", None)
            if close is not None:
                close()
            self._session = None


def create_snowflake_session(settings: ClaimKbSettings):
    from snowflake.snowpark import Session

    return (
        Session.builder.config(
            "connection_name",
            settings.snowflake_connection_name,
        ).create()
    )


def _row_get(row: Any, name: str, position: int) -> Any:
    candidates = (name, name.upper(), name.lower())
    for candidate in candidates:
        try:
            return row[candidate]
        except (KeyError, TypeError, AttributeError):
            pass
    return row[position]


def _coerce_embedding(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]
