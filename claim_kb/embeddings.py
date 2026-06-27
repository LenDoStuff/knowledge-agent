"""Snowflake Cortex embedding adapter."""

from __future__ import annotations

from typing import Protocol, Sequence

class TextEmbedder(Protocol):
    embedding_provider: str
    embedding_model: str

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def close(self) -> None:
        ...


class SnowflakeAiEmbedder:
    embedding_provider = "snowflake"

    def __init__(self, connection_name: str, embedding_model: str) -> None:
        self.embedding_model = embedding_model
        self._session = create_snowflake_session(connection_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        from snowflake.snowpark.functions import ai_embed, col

        rows = [(index, text) for index, text in enumerate(texts)]
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
            index = int(row["CHUNK_INDEX"])
            embeddings_by_index[index] = _coerce_embedding(row["EMBEDDING"])
        return [embeddings_by_index[index] for index in range(len(texts))]

    def close(self) -> None:
        self._session.close()


def create_snowflake_session(connection_name: str):
    from snowflake.snowpark import Session

    return (
        Session.builder.config(
            "connection_name",
            connection_name,
        ).create()
    )


def _coerce_embedding(value: Sequence[float]) -> list[float]:
    return [float(item) for item in value]
