"""Every module under src/ must import under CI's dependency set.

CI installs a subset of requirements.txt to keep the run fast. A third-party
import added to src/ without being added to that list fails only in CI, and
only in whichever test happens to touch the module first. This makes that
failure name itself.

The module list is **discovered, not written down**. A hand-maintained list
drifts: when this file first shipped it named the modules that existed that
day, and the next two modules added to src/ were never added to it, so the
guard silently stopped covering them. Walking the package cannot drift.

src.encoders is excluded on purpose: it is the one module that needs
transformers, and the feature-extraction path that uses it is not under test.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Needs transformers, which CI deliberately does not install.
EXCLUDED = {"src.encoders"}


def discover() -> list[str]:
    import src

    found = [
        name
        for _, name, _ in pkgutil.walk_packages(src.__path__, prefix="src.")
        if name not in EXCLUDED
    ]
    return ["src", *sorted(found)]


MODULES = discover()


def test_discovery_found_the_package() -> None:
    """A broken walk would make every other test here vacuously pass."""
    assert len(MODULES) > 5, MODULES
    assert "src.config" in MODULES
    assert "src.encoders" not in MODULES


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
