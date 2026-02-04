import json
import os
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fal_client

from config import DATA_DIR
from io_utils import build_data_uri


def parse_optional_int(value: str) -> Optional[int]:
    """Parse an optional integer from user input."""
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError("Seed must be an integer.") from exc


def normalize_fal_result(result: Any) -> Dict[str, Any]:
    """Normalize fal.ai result objects into a dict."""
    if isinstance(result, dict):
        return result
    if hasattr(result, "data"):
        payload: Dict[str, Any] = {"data": result.data}
        if hasattr(result, "request_id"):
            payload["request_id"] = result.request_id
        return payload
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    return {"result": repr(result)}


def save_json_debug(data: Dict[str, Any], job_dir: Path) -> str:
    """Persist the raw fal.ai response for debugging."""
    outputs_dir = job_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    target_path = outputs_dir / "response.json"
    target_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return str(target_path.relative_to(DATA_DIR))


def run_fal_instruct_edit(
    model_id: str,
    prompt: str,
    input_images: List[Dict[str, Any]],
    image_size: str,
    num_images: int,
    guidance_scale: float,
    seed: Optional[int],
    enable_safety_checker: bool,
    output_format: str,
    job_dir: Path,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Run fal.ai instruct edit and store returned images."""
    fal_key = os.environ.get("FAL_KEY", "")
    if not fal_key:
        raise RuntimeError("FAL_KEY environment variable is required for fal.ai requests.")

    image_urls = []
    for image in input_images:
        image_path = DATA_DIR / image["path"]
        image_urls.append(build_data_uri(image_path, image["mime_type"]))

    payload = {
        "prompt": prompt,
        "image_urls": image_urls,
        "image_size": image_size,
        "num_images": int(num_images),
        "guidance_scale": float(guidance_scale),
        "enable_safety_checker": bool(enable_safety_checker),
        "output_format": output_format,
    }
    if seed is not None:
        payload["seed"] = int(seed)

    def on_queue_update(update: Any) -> None:
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(log.get("message", ""))

    result = fal_client.subscribe(
        model_id,
        arguments=payload,
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    response_data = normalize_fal_result(result)

    debug_path = save_json_debug(response_data, job_dir)
    outputs_dir = job_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    result_payload = response_data.get("data", response_data)
    output_images: List[Dict[str, Any]] = []
    for idx, image in enumerate(result_payload.get("images", []), start=1):
        url = image.get("url")
        if not url:
            continue
        mime = image.get("content_type") or "image/png"
        ext = mime.split("/")[-1] or "png"
        filename = f"output_{idx}.{ext}"
        target_path = outputs_dir / filename
        img_response = requests.get(url, timeout=60)
        img_response.raise_for_status()
        target_path.write_bytes(img_response.content)
        output_images.append(
            {
                "path": str(target_path.relative_to(DATA_DIR)),
                "mime_type": mime,
                "source_url": url,
            }
        )

    return "", output_images, debug_path
