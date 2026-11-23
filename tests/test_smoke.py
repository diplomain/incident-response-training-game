"""Smoke tests checking core system plumbing and CLI integration."""
from app.storage import Store
from app.cli import main
from app.core import milestone_1, NextEventModel


def test_the_store_keeps_what_you_put_in_it(tmp_path):
    store = Store(str(tmp_path / "app.json"))
    store.put("hello", {"n": 1})
    assert store.get("hello") == {"n": 1}
    assert store.get("missing", "fallback") == "fallback"
    assert store.all() == {"hello": {"n": 1}}


def test_the_command_line_runs(capsys):
    main(["hello"])
    assert "is ready" in capsys.readouterr().out


def test_core_smoke_pipeline():
    incidents = milestone_1()
    assert len(incidents) >= 20
    model = NextEventModel().train()
    preds = model.predict(incidents[0][:2], top_k=2)
    assert len(preds) > 0
