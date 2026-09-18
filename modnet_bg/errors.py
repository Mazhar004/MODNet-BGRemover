"""Exception hierarchy shared across the package.

Every error surfaced to a user is one of these. The web layer maps each
to an HTTP status; nothing else is allowed to reach a client.
"""


class BGRemoverError(Exception):
    """Base class for every error this package raises deliberately."""


class UnsupportedMediaError(BGRemoverError):
    """Upload is not an image or video format we accept. -> HTTP 415"""


class MediaTooLargeError(BGRemoverError):
    """Upload exceeds a configured byte, pixel, or duration cap. -> HTTP 413"""


class CorruptMediaError(BGRemoverError):
    """Upload has an accepted type but cannot be decoded. -> HTTP 422"""


class WeightsError(BGRemoverError):
    """Checkpoint is missing, unreachable, or fails its digest check."""


class JobNotFoundError(BGRemoverError):
    """No job with that id, or it has passed its TTL. -> HTTP 404"""


class JobFailedError(BGRemoverError):
    """Processing raised. The message is safe to show a user."""
