from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

from config import EXAMPLES_DIR
from io_utils import guess_mime


def load_example_inputs() -> List[Dict[str, Any]]:
    """Load prompt and images from the examples directory."""
    examples: List[Dict[str, Any]] = []
    if not EXAMPLES_DIR.exists():
        return examples
    for child in sorted(EXAMPLES_DIR.iterdir()):
        if not child.is_dir():
            continue
        prompt_path = child / "prompt.txt"
        prompt_text = ""
        if prompt_path.exists():
            prompt_text = prompt_path.read_text(encoding="utf-8")

        def image_sort_key(path: Path) -> Tuple[int, int, str]:
            stem = path.stem.strip()
            if stem.isdigit():
                return (0, int(stem), path.name.lower())
            return (1, 0, path.name.lower())

        image_paths = [
            path
            for path in sorted(child.iterdir(), key=image_sort_key)
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        images = [
            {
                "source_path": str(path),
                "filename": path.name,
                "mime_type": guess_mime(path.name, "image/png"),
            }
            for path in image_paths
        ]
        examples.append(
            {
                "id": child.name,
                "label": child.name,
                "prompt": prompt_text,
                "images": images,
            }
        )
    return examples


def apply_example_input(example: Dict[str, Any]) -> None:
    """Apply an example prompt and images to session state."""
    st.session_state["prompt_text"] = example.get("prompt", "")
    st.session_state["example_images"] = example.get("images", [])
    st.session_state["example_label"] = example.get("label", "")
