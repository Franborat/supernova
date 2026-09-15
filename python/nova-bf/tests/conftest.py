"""Shared fixtures for the nova-bf test suite."""

from __future__ import annotations

import pytest

_MERGE_ENV = ("NOVA_BF_MERGE_FORCE", "NOVA_BF_MERGE_WINDOW")


@pytest.fixture(autouse=True)
def _merge_env_is_hermetic(monkeypatch):
    """No test inherits the operator's merge overrides.

    `NOVA_BF_MERGE_FORCE` is meant to be EXPORTED for a rescue run, so it will
    be set in the shell of whoever is working on this code -- exactly when
    these tests get run. Exported, it turns six of them red for the wrong
    reason and, more quietly, makes every other merge test in the repo
    exercise the forced path while still reporting green. Tests that want it
    set it themselves.
    """
    for name in _MERGE_ENV:
        monkeypatch.delenv(name, raising=False)
