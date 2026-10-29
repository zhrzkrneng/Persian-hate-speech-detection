"""Device selection.

Kept out of `encoders` so that training, which only ever reads cached features,
does not have to import transformers to pick a device.
"""

from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)
