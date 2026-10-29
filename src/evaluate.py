"""Evaluation over a cached-feature split."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils.metrics import Metrics, compute_metrics


@torch.no_grad()
def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, P(hateful))."""
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for image_features, text_features, batch_labels in loader:
        logits = model(image_features.to(device), text_features.to(device))
        positive = torch.softmax(logits, dim=-1)[:, 1]
        probabilities.append(positive.cpu().numpy())
        labels.append(batch_labels.numpy())

    return np.concatenate(labels), np.concatenate(probabilities)


def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Metrics:
    labels, probabilities = predict(model, loader, device)
    return compute_metrics(labels, probabilities)
