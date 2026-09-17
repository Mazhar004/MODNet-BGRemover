"""Which compute device to run on, and how a user preference maps onto it.

Preference resolution never fails. An unavailable or unknown request
clamps down to the best device that does exist, and the caller reports
what was actually used rather than erroring.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

DEVICE_PREFERENCES: tuple[str, ...] = ("auto", "cuda", "mps", "cpu")

# Best first. available_devices() filters this; resolve_device() scans it.
_PRIORITY: tuple[str, ...] = ("cuda", "mps", "cpu")


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def available_devices() -> list[str]:
    """Devices this process can actually use, best first. Always includes cpu."""
    present = {"cpu": True, "cuda": _cuda_available(), "mps": _mps_available()}
    return [name for name in _PRIORITY if present[name]]


def resolve_device(pref: str | None) -> str:
    """Map a preference onto a real device. Never raises."""
    devices = available_devices()
    normalised = (pref or "auto").strip().lower()

    if normalised in devices:
        return normalised

    if normalised not in DEVICE_PREFERENCES:
        logger.warning("Unknown device preference %r; using auto", pref)
    elif normalised != "auto":
        logger.info("Device %r unavailable; falling back to %s", normalised, devices[0])

    return devices[0]
