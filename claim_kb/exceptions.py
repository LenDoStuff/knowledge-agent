"""Domain exceptions for claim knowledge base operations."""


class ClaimKbError(Exception):
    """Base exception for claim knowledge base failures."""


class ConfigurationError(ClaimKbError):
    """Raised when required runtime configuration is missing or invalid."""


class ClaimNotFoundError(ClaimKbError):
    """Raised when a claim folder cannot be found."""


class DocumentNotFoundError(ClaimKbError):
    """Raised when a document cannot be found within a claim."""


class ChunkNotFoundError(ClaimKbError):
    """Raised when a chunk cannot be found within a claim document."""


class IngestionError(ClaimKbError):
    """Raised when ingestion cannot complete."""
