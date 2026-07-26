"""Import-smoke test for the ported visualization modules.

Proves ``dlofeel_state.viz.predictions`` and ``dlofeel_state.viz.dataset``
import cleanly against the ported package (correct rewired imports) and that
each exposes a callable ``main`` entrypoint. Display code (viser) is never
exercised: importing the module does not construct a ViserServer, so no viewer
is launched and no display is required.

Run::

    pytest tests/test_viz_smoke.py -v
"""

import importlib


def test_predictions_imports_and_has_main():
    mod = importlib.import_module("dlofeel_state.viz.predictions")
    assert callable(mod.main)


def test_dataset_viz_imports_and_has_main():
    mod = importlib.import_module("dlofeel_state.viz.dataset")
    assert callable(mod.main)
