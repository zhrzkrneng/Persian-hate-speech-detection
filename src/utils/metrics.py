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
    """Mean and sample standard deviation of each metric across seeds.

    Sample, not population: ddof=1. Five seeds are a sample of the random
    initialisations and data orders the method could have been run under, not
    the population of them, so the unbiased estimate is the one that answers
    "how much does this number move if I run it again".

    The difference is not cosmetic at this n. With five seeds, ddof=0 reports
    a spread sqrt(4/5) = 0.894 times the sample one, about 11% narrower, and it
    is narrower in the direction that flatters: the comparisons this project
    exists to make differ by well under a point of AUROC, so understating the
    spread is what would make a difference look resolved when it is not.

    A single run has no spread to estimate, and nan is the honest answer -- a
    0.00 there would read as a result that never varies.
    """
    if not runs:
        raise ValueError("no runs to summarise")
    summary = {}
    for key in ("auroc", "accuracy", "macro_f1"):
        values = np.array([getattr(run, key) for run in runs], dtype=float)
        spread = float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan")
        summary[key] = (float(np.nanmean(values)), spread)
    return summary
