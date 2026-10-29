"""Inter-annotator agreement.

Krippendorff's alpha is what the annotation guideline commits to reporting, and
it is easy to implement subtly wrong: a wrong alpha is a plausible-looking
number that nothing downstream contradicts. So this follows Krippendorff's
own formulation and is checked in tests against his published worked example.

Definition, for nominal data:

    alpha = 1 - Do / De

where Do is observed disagreement and De the disagreement expected by chance.
alpha = 1 is perfect agreement, 0 is chance, and negative values mean
systematic disagreement — annotators doing worse than coin flips, which
usually points at a guideline two people read differently.

Units coded by fewer than two annotators carry no information about agreement
and are excluded, as the method requires.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Sequence


def krippendorff_alpha(
    units: Iterable[Sequence[Hashable | None]],
) -> float:
    """Nominal-scale alpha.

    `units` is one sequence per item, holding each annotator's label for it.
    `None` marks a label that annotator did not give — an abstention, or an
    item they never saw.

    Raises ValueError when no unit has at least two labels, since alpha is
    undefined there rather than zero.
    """
    # The coincidence matrix counts each pair of labels within a unit, weighted
    # by 1/(m_u - 1) so that a unit coded by many annotators does not dominate
    # one coded by few.
    coincidences: Counter[tuple[Hashable, Hashable]] = Counter()
    usable = 0

    for unit in units:
        labels = [label for label in unit if label is not None]
        count = len(labels)
        if count < 2:
            continue
        usable += 1
        weight = 1.0 / (count - 1)
        for i, first in enumerate(labels):
            for j, second in enumerate(labels):
                if i != j:
                    coincidences[(first, second)] += weight

    if usable == 0:
        raise ValueError("no unit has two or more labels; alpha is undefined, not zero")

    total = sum(coincidences.values())
    if total == 0:
        raise ValueError("coincidence matrix is empty")

    # Marginals: how often each label value occurs across the matrix.
    marginals: Counter[Hashable] = Counter()
    for (first, _), value in coincidences.items():
        marginals[first] += value

    # Observed disagreement: mass off the diagonal.
    agreeing = sum(coincidences[(v, v)] for v in marginals)
    observed = (total - agreeing) / total

    # Expected disagreement under independent sampling without replacement.
    expected_agreeing = sum(n * (n - 1) for n in marginals.values())
    expected = 1.0 - expected_agreeing / (total * (total - 1))

    if expected == 0:
        # Every annotator used one label everywhere. Agreement is then
        # trivially perfect and alpha carries no information.
        return 1.0

    return 1.0 - observed / expected


def pairwise_agreement(units: Iterable[Sequence[Hashable | None]]) -> float:
    """Raw proportion of agreeing annotator pairs, ignoring chance.

    Reported alongside alpha because it is what Multi3Hate quotes when it says
    annotators from different countries agreed 74% of the time. It is easier
    to read and systematically more flattering, so it never stands alone.
    """
    agree = 0
    pairs = 0
    for unit in units:
        labels = [label for label in unit if label is not None]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs += 1
                agree += labels[i] == labels[j]
    if pairs == 0:
        raise ValueError("no annotator pairs to compare")
    return agree / pairs
