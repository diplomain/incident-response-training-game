"""Unit tests for the key-value Store (app/storage.py)."""
from pathlib import Path
from app.storage import Store


def test_store_operations(tmp_path: Path):
    db_file = tmp_path / "test_store.json"
    store = Store(str(db_file))
    
    assert store.get("missing", "default_val") == "default_val"
    
    store.put("leaderboard", [{"user": "pilot", "score": 85.5}])
    assert len(store.get("leaderboard")) == 1
    assert store.get("leaderboard")[0]["score"] == 85.5
    
    all_data = store.all()
    assert "leaderboard" in all_data
