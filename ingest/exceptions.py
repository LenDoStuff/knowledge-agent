"""Domain exceptions for claim knowledge base operations."""

from infrastructure.config import ConfigurationError


class ClaimKbError(Exception):
    """Base exception for claim knowledge base failures."""


class ClaimNotFoundError(ClaimKbError):
    """Raised when a claim folder cannot be found."""


class DocumentNotFoundError(ClaimKbError):
    """Raised when a document cannot be found within a claim."""


class PageNotFoundError(ClaimKbError):
    """Raised when an OCR page cannot be found within a claim."""


class ChunkNotFoundError(ClaimKbError):
    """Raised when a chunk cannot be found within a claim document."""
