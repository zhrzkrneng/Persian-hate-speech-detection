"""Cross-modal fusion variants.

Given projected image features `p_i` and text features `p_t`, both of width n:

  concat  [p_i ; p_t]                      -> 2n
  align   diag(p_i (x) p_t) = p_i * p_t    -> n
  cross   flatten(p_i (x) p_t)             -> n^2

`align` keeps only the diagonal of the feature interaction matrix. The sum of
those diagonal elements is the dot product between the two vectors, so the
vector itself is a per-dimension view of how well the modalities agree. It costs
n features instead of n^2, which is why it is the default here: at n = 1024 the
cross variant puts roughly a billion parameters in the first classifier layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

FUSIONS = ("concat", "align", "cross")


def fused_dim(fusion: str, projection_dim: int) -> int:
    if fusion == "concat":
        return 2 * projection_dim
    if fusion == "align":
        return projection_dim
    if fusion == "cross":
        return projection_dim**2
    raise ValueError(f"unknown fusion {fusion!r}, expected one of {FUSIONS}")


class CrossModalFusion(nn.Module):
    def __init__(self, fusion: str) -> None:
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"unknown fusion {fusion!r}, expected one of {FUSIONS}")
        self.fusion = fusion

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if image.shape != text.shape:
            raise ValueError(
                f"projected features must match: {tuple(image.shape)} vs "
                f"{tuple(text.shape)}"
            )
        if self.fusion == "concat":
            return torch.cat([image, text], dim=-1)
        if self.fusion == "align":
            return image * text
        return torch.einsum("bi,bj->bij", image, text).flatten(start_dim=1)

    def extra_repr(self) -> str:
        return f"fusion={self.fusion}"
