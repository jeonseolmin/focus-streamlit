"""Smoke test: the dumped model loads, answers clear-cut cases, and the page renders. Run: python test_app.py"""
from streamlit.testing.v1 import AppTest

from app import history, load_ml_objects, predict


def state(**kw):
    args = dict(early_app="ide", recent_app="ide", switches=2, idle_min=0, weekday=1, hour=14) | kw
    proba, x = predict(*load_ml_objects(), history(**args))
    assert x.shape == (1, 28) and abs(proba.sum() - 1) < 1e-6
    return proba.idxmax()


def test_cases():
    assert state() == "focus"
    assert state(early_app="game", recent_app="game") == "distracted"
    assert state(idle_min=15) == "break"


def test_page():
    at = AppTest.from_file("app.py").run(timeout=60)
    assert not at.exception and len(at.tabs) == 5
    at.button[0].click().run(timeout=60)
    assert not at.exception and at.metric[0].value == "집중"


if __name__ == "__main__":
    test_cases()
    test_page()
    print("ok")
