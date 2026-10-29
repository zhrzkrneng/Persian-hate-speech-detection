"""Evaluation metrics.

AUROC is the headline metric on the Hateful Memes Challenge, so it is what the
numbers in this repository are comparable on. Accuracy and macro F1 are kept
alongside it because the Persian set will be small and class-imbalanced, where
AUROC alone hides a lot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass
class Metrics:
    auroc: float
    accuracy: float
    macro_f1: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"AUROC {self.auroc * 100:.2f} | "
            f"acc {self.accuracy * 100:.2f} | "
            f"macro-F1 {self.macro_f1 * 100:.2f}"
        )


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Metrics:
    """`probabilities` is P(hateful) for each example."""
    predictions = (probabilities >= 0.5).astype(int)

    # A split with a single class present makes AUROC undefined. That happens
    # on tiny evaluation sets, which the Persian subset will be.
    if len(np.unique(labels)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(labels, probabilities))

    return Metrics(
        auroc=auroc,
        accuracy=float(accuracy_score(labels, predictions)),
        macro_f1=float(f1_score(labels, predictions, average="macro")),
    )


def summarise(runs: list[Metrics]) -> dict[str, tuple[float, float]]:
    """Mean and population standard deviation of each metric across seeds."""
    if not runs:
        raise ValueError("no runs to summarise")
    summary = {}
    for key in ("auroc", "accuracy", "macro_f1"):
        values = np.array([getattr(run, key) for run in runs], dtype=float)
        summary[key] = (float(np.nanmean(values)), float(np.nanstd(values)))
    return summary
