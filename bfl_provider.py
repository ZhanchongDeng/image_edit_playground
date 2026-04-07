import base64
import json
import os
import time
import urllib3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import BFL_MODELS, DATA_DIR
from fal_provider import get_image_dimensions


def _bfl_requests_verify() -> bool:
    """BFL Labs staging often uses a cert that fails normal verification (official examples use curl -k).

    Default: do not verify TLS. Set ``BFL_SSL_VERIFY=1`` (or ``true``) to verify certificates.
    """
    return os.environ.get("BFL_SSL_VERIFY", "").strip().lower() in ("1", "true", "yes")


def _image_to_base64(image: Dict[str, Any]) -> str:
    if image.get("url"):
        resp = requests.get(image["url"], timeout=60)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("ascii")
    path = DATA_DIR / image["path"]
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _resolve_dimensions(
    image_size: Optional[str],
    input_images: List[Dict[str, Any]],
) -> Tuple[int, int]:
    if input_images:
        dims = get_image_dimensions(input_images[0])
        if dims:
            return dims
    preset: Dict[str, Tuple[int, int]] = {
        "auto": (1024, 1024),
        "square_hd": (1024, 1024),
        "square": (1024, 1024),
        "portrait_4_3": (896, 1152),
        "portrait_16_9": (720, 1280),
        "landscape_4_3": (1152, 896),
        "landscape_16_9": (1280, 720),
    }
    return preset.get(image_size or "auto", (1024, 1024))


def _scale_to_max_side(width: int, height: int, max_side: int) -> Tuple[int, int]:
    """Uniform scale so the longer edge is at most ``max_side`` (FLUX.2 Pro 2k cap)."""
    if width <= 0 or height <= 0 or max_side <= 0:
        return width, height
    m = max(width, height)
    if m <= max_side:
        return width, height
    scale = max_side / m
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _input_image_json_key(index: int) -> str:
    """BFL uses input_image, input_image_2 … input_image_10 (FLUX.2 Pro contract)."""
    if index == 0:
        return "input_image"
    if 1 <= index <= 9:
        return f"input_image_{index + 1}"
    raise ValueError(f"input_image index out of range (0–9): {index}")


def build_bfl_payload(
    model_id: str,
    prompt: str,
    input_images: List[Dict[str, Any]],
    image_size: Optional[str],
    seed: Optional[int],
    prompt_upsampling: bool,
    safety_tolerance: str,
) -> Dict[str, Any]:
    model_config = BFL_MODELS[model_id]
    payload_type = model_config.get("payload_type", "flux2_pro_partner")
    width, height = _resolve_dimensions(image_size, input_images)
    max_side = int(model_config.get("max_output_side") or 0)
    if max_side > 0:
        width, height = _scale_to_max_side(width, height, max_side)

    if payload_type == "flux2_pro_partner":
        min_im = int(model_config.get("min_input_images") or 1)
        max_im = int(model_config.get("max_input_images") or 10)
        n = len(input_images)
        if n < min_im:
            raise ValueError(f"This model requires at least {min_im} input images.")
        if n > max_im:
            raise ValueError(f"This model supports at most {max_im} input images.")
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "prompt_upsampling": bool(prompt_upsampling),
            "safety_tolerance": int(safety_tolerance),
        }
        for i in range(n):
            payload[_input_image_json_key(i)] = _image_to_base64(input_images[i])
        if seed is not None:
            payload["seed"] = int(seed)
        return payload

    raise ValueError(f"Unsupported BFL payload_type: {payload_type}")


def save_json_debug(data: Dict[str, Any], job_dir: Path) -> str:
    outputs_dir = job_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    target_path = outputs_dir / "response.json"
    target_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return str(target_path.relative_to(DATA_DIR))


def run_bfl_request(
    model_id: str,
    prompt: str,
    input_images: List[Dict[str, Any]],
    image_size: Optional[str],
    seed: Optional[int],
    prompt_upsampling: bool,
    safety_tolerance: str,
    job_dir: Path,
    poll_interval_s: float = 1.0,
    poll_timeout_s: float = 600.0,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Submit to BFL Labs partner API, poll until ready, download the output image."""
    api_key = os.environ.get("BFL_API_KEY", "")
    if not api_key:
        raise RuntimeError("BFL_API_KEY environment variable is required for BFL Labs requests.")

    model_config = BFL_MODELS[model_id]
    base_url = model_config.get("base_url", "https://labs.us2.bfl.ai").rstrip("/")
    submit_path = model_config.get("submit_path", "/partners/raspberryai/flux-2-pro")
    if not submit_path.startswith("/"):
        submit_path = "/" + submit_path

    payload = build_bfl_payload(
        model_id=model_id,
        prompt=prompt,
        input_images=input_images,
        image_size=image_size,
        seed=seed,
        prompt_upsampling=prompt_upsampling,
        safety_tolerance=safety_tolerance,
    )

    outputs_dir = job_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Full outgoing JSON (large — includes base64 images) for manual inspection.
    request_body_path = outputs_dir / "request_body.json"
    request_body_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    verify = _bfl_requests_verify()
    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    headers = {"X-Key": api_key, "Content-Type": "application/json"}
    submit_url = f"{base_url}{submit_path}"

    request_meta_path = outputs_dir / "request_meta.json"
    request_meta_path.write_text(
        json.dumps(
            {
                "method": "POST",
                "url": submit_url,
                "headers": {"Content-Type": "application/json", "X-Key": "<redacted>"},
                "verify_ssl": verify,
                "request_body_path": "request_body.json",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    submit_resp = requests.post(
        submit_url, headers=headers, json=payload, timeout=120, verify=verify
    )
    submit_resp.raise_for_status()
    submit_data = submit_resp.json()

    task_id = submit_data.get("id")
    if not task_id:
        raise RuntimeError(f"BFL submit response missing id: {submit_data!r}")

    poll_url = submit_data.get("polling_url") or f"{base_url}/v1/get_result?id={task_id}"

    params_path = outputs_dir / "params.json"
    # Omit huge base64 blobs from params dump
    params_redacted = {
        k: v for k, v in payload.items() if k != "input_image" and not k.startswith("input_image_")
    }
    params_redacted["_note"] = "input_image … input_image_10 omitted (base64)"
    params_path.write_text(json.dumps(params_redacted, indent=2, sort_keys=True), encoding="utf-8")

    deadline = time.time() + poll_timeout_s
    last_result: Dict[str, Any] = {}
    while time.time() < deadline:
        poll_resp = requests.get(
            poll_url, headers={"X-Key": api_key}, timeout=60, verify=verify
        )
        poll_resp.raise_for_status()
        last_result = poll_resp.json()
        status = last_result.get("status")
        if status == "Ready":
            break
        if status in ("Error", "Request Moderated", "Content Moderated"):
            raise RuntimeError(f"BFL job failed: {status} — {last_result!r}")
        time.sleep(poll_interval_s)
    else:
        raise RuntimeError(f"BFL polling timed out after {poll_timeout_s}s. Last: {last_result!r}")

    debug_path = save_json_debug(last_result, job_dir)

    result_block = last_result.get("result") or {}
    sample_url = result_block.get("sample")
    if not sample_url:
        raise RuntimeError(f"BFL result missing result.sample: {last_result!r}")

    img_response = requests.get(sample_url, timeout=120, verify=verify)
    img_response.raise_for_status()
    mime = img_response.headers.get("Content-Type", "image/jpeg").split(";")[0].strip() or "image/jpeg"
    ext = mime.split("/")[-1] or "jpg"
    filename = f"output_1.{ext}"
    target_path = outputs_dir / filename
    target_path.write_bytes(img_response.content)
    output_images = [
        {
            "path": str(target_path.relative_to(DATA_DIR)),
            "mime_type": mime,
            "source_url": sample_url,
        }
    ]
    output_text = ""
    return output_text, output_images, debug_path
