"""Every module under src/ and scripts/ must import under CI's dependency set.

CI installs a subset of requirements.txt to keep the run fast. A third-party
import added without being added to that list fails only in CI, and only in
whichever test happens to touch the module first. This makes that failure name
itself.

The module list is **discovered, not written down**. A hand-maintained list
drifts: when this file first shipped it named the modules that existed that
day, and the next two modules added to src/ were never added to it, so the
guard silently stopped covering them. Walking the package cannot drift.

scripts/ is covered as well as src/, and was not at first. That hole showed
itself when scripts/compare_cells.py began importing scipy, which CI had never
installed and which only arrived because scikit-learn happens to depend on it.
Every script guards its work behind __main__, so importing one runs no work.

Modules that need transformers are excluded on purpose: it is a multi-gigabyte
download CI does not do, and the feature-extraction path that uses it is not
under test.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Need transformers, which CI deliberately does not install.
EXCLUDED = {"src.encoders", "scripts.extract_features", "scripts.eval_retrieval"}


def discover() -> list[str]:
    import scripts
    import src

    found = []
    for package, prefix in ((src, "src."), (scripts, "scripts.")):
        found += [
            name
            for _, name, _ in pkgutil.walk_packages(package.__path__, prefix=prefix)
            if name not in EXCLUDED
        ]
    return ["src", *sorted(found)]


MODULES = discover()


def test_discovery_found_the_packages() -> None:
    """A broken walk would make every other test here vacuously pass."""
    assert len(MODULES) > 5, MODULES
    assert "src.config" in MODULES
    assert "scripts.compare_cells" in MODULES, "scripts/ must be covered too"
    assert "src.encoders" not in MODULES


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
