"""Claim storage errors."""


class ClaimError(Exception):
    """Base error for claim lookup failures."""


class DocumentNotFoundError(ClaimError):
    pass


class PageNotFoundError(ClaimError):
    pass


class ChunkNotFoundError(ClaimError):
    pass
