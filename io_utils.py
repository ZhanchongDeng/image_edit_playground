import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from config import DATA_DIR


def guess_mime(filename: str, default: str = "application/octet-stream") -> str:
    """Infer MIME type from a filename, falling back to default."""
    mime, _ = mimetypes.guess_type(filename)
    return mime or default


def save_uploaded_images(
    job_dir: Path,
    uploads: List[st.runtime.uploaded_file_manager.UploadedFile],
) -> List[Dict[str, Any]]:
    """Persist uploaded images to the job input directory."""
    inputs_dir = job_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for upload in uploads:
        filename = upload.name
        mime = upload.type or guess_mime(filename, "image/png")
        payload = upload.getvalue()
        target_path = inputs_dir / filename
        target_path.write_bytes(payload)
        saved.append(
            {
                "filename": filename,
                "path": str(target_path.relative_to(DATA_DIR)),
                "mime_type": mime,
                "size_bytes": len(payload),
            }
        )
    return saved


def save_example_images(job_dir: Path, example_images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy example images into the job input directory."""
    inputs_dir = job_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for image in example_images:
        source_path = Path(image["source_path"])
        if not source_path.exists():
            continue
        target_path = inputs_dir / image["filename"]
        target_path.write_bytes(source_path.read_bytes())
        saved.append(
            {
                "filename": image["filename"],
                "path": str(target_path.relative_to(DATA_DIR)),
                "mime_type": image["mime_type"],
                "size_bytes": target_path.stat().st_size,
            }
        )
    return saved


def build_data_uri(image_path: Path, mime_type: str) -> str:
    """Encode an image file as a data URI string."""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
