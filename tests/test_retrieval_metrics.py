"""The retrieval metric decides which encoder the project builds on.

If it is wrong, every number in the comparison is wrong in the same direction
and nothing downstream would reveal it. So it is checked against cases whose
answer is known by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_retrieval import evaluate, recall_at_k  # noqa: E402


def test_recall_at_k_counts_ranks() -> None:
    ranks = torch.tensor([0, 1, 4, 9, 50])
    scores = recall_at_k(ranks, (1, 5, 10))
    # float32 means these land a few ULPs off the exact percentage
    assert scores["R@1"] == pytest.approx(20.0)  # one rank-0
    assert scores["R@5"] == pytest.approx(60.0)  # ranks 0, 1, 4
    assert scores["R@10"] == pytest.approx(80.0)  # plus rank 9


def test_perfect_retrieval_scores_100() -> None:
    """Identical image and text features: the right answer always ranks first."""
    features = torch.eye(6)
    mapping = torch.arange(6)
    results = evaluate(features, features, mapping)
    for direction in ("text_to_image", "image_to_text"):
        assert results[direction]["R@1"] == 100.0


def test_worst_case_scores_zero_at_one() -> None:
    """Every caption is most similar to the wrong image, by construction."""
    images = torch.eye(4)
    # caption i points exactly at image i+1 (mod 4), never its own
    texts = torch.roll(torch.eye(4), shifts=1, dims=1)
    results = evaluate(images, texts, torch.arange(4))
    assert results["text_to_image"]["R@1"] == 0.0


def test_multiple_captions_per_image() -> None:
    """Two captions per image; a hit needs only one of them to rank."""
    images = torch.eye(3)
    texts = torch.cat([torch.eye(3), torch.eye(3)])
    mapping = torch.tensor([0, 1, 2, 0, 1, 2])
    results = evaluate(images, texts, mapping)
    assert results["text_to_image"]["R@1"] == 100.0
    assert results["image_to_text"]["R@1"] == 100.0


def test_random_features_score_near_chance() -> None:
    """Unrelated features must not look like retrieval working."""
    torch.manual_seed(0)
    n = 200
    results = evaluate(torch.randn(n, 64), torch.randn(n, 64), torch.arange(n))
    # chance R@1 is 100/200 = 0.5%; allow headroom without allowing a real score
    assert results["text_to_image"]["R@1"] < 5.0


def test_scale_invariance() -> None:
    """Features are normalised, so magnitude must not change the ranking."""
    torch.manual_seed(0)
    images, texts = torch.randn(10, 16), torch.randn(10, 16)
    mapping = torch.arange(10)
    assert evaluate(images, texts, mapping) == evaluate(
        images * 13.0, texts * 0.07, mapping
    )
