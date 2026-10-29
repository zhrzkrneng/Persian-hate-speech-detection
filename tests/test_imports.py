"""Every module the tests can reach must import under CI's dependency set.

CI installs a subset of requirements.txt (no transformers, pillow or pandas)
to keep the run fast. A third-party import added to src/ without being added
to that list fails only in CI, and only in whichever test happens to touch the
module first. This test makes that failure name itself.

src.encoders is excluded on purpose: it is the one module that needs
transformers, and the feature-extraction path that uses it is not under test.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = [
    "src.config",
    "src.data",
    "src.data.features",
    "src.data.hmc",
    "src.evaluate",
    "src.models",
    "src.models.classifier",
    "src.models.fusion",
    "src.models.memeclip",
    "src.train",
    "src.utils",
    "src.utils.device",
    "src.utils.metrics",
    "src.utils.seed",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
