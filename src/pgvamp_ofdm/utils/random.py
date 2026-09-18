"""Versioned, order-independent random streams (never Python hash())."""

import hashlib
import json
import random
from typing import Any

import numpy as np
import torch

SEED_VERSION = "sha256-json-v1"


def stable_hash(value: Any) -> str:
    """Hash canonical, finite JSON using UTF-8."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def derive_seed(master: int, stream: str, *identity: Any) -> int:
    """Separate domains and identities without consuming another stream's draws."""
    if type(master) is not int or master < 0 or not isinstance(stream, str) or not stream:
        raise ValueError("master seed and stream must be nonnegative integer / nonempty string")
    return int(stable_hash([SEED_VERSION, master, stream, identity])[:16], 16) % (2**63)


def generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def seed_runtime(seed: int, device: torch.device) -> None:
    """Initialize external RNGs; samples still use explicit independent generators."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
