"""Smoke test: the dumped model loads and answers clear-cut cases. Run: python test_app.py (or pytest)."""
from app import history, load_model, predict


def state(**kw):
    args = dict(early_app="ide", recent_app="ide", switches=2, idle_min=0, weekday=1, hour=14) | kw
    proba, x = predict(load_model(), history(**args))
    assert x.shape == (1, 28) and abs(proba.sum() - 1) < 1e-6
    return proba.idxmax()


def test_cases():
    assert state() == "focus"
    assert state(early_app="game", recent_app="game") == "distracted"
    assert state(idle_min=15) == "break"


if __name__ == "__main__":
    test_cases()
    print("ok")
