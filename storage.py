import json
from pathlib import Path
from typing import Any, Dict, List


DATA_DIR = Path(__file__).resolve().parent / "data"
HISTORY_PATH = DATA_DIR / "history.json"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_history() -> List[Dict[str, Any]]:
    ensure_data_dir()
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return []


def save_history(history: List[Dict[str, Any]]) -> None:
    ensure_data_dir()
    with HISTORY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def append_job(job: Dict[str, Any]) -> None:
    history = load_history()
    history.insert(0, job)
    save_history(history)


def update_job(job_id: str, updates: Dict[str, Any]) -> None:
    history = load_history()
    for job in history:
        if job.get("id") == job_id:
            job.update(updates)
            break
    save_history(history)

