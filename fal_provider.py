import json
import os
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fal_client
from PIL import Image

from config import DATA_DIR, FAL_MODELS
from io_utils import build_data_uri


def get_image_dimensions(image: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Return (width, height) of the first image, from local path or URL."""
    try:
        if image.get("url"):
            resp = requests.get(image["url"], timeout=30, stream=True)
            resp.raise_for_status()
            img = Image.open(resp.raw)
        else:
            path = DATA_DIR / image["path"]
            if not path.exists():
                return None
            img = Image.open(path)
        img.load()
        return (img.width, img.height)
    except Exception:
        return None


def upload_local_images_to_fal(
    input_images: List[Dict[str, Any]],
    fal_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Attach fal CDN URLs to locally stored images. No-op without a key."""
    key = (fal_key or "").strip() or os.environ.get("FAL_KEY", "").strip()
    if not key:
        return input_images
    prev = os.environ.get("FAL_KEY")
    os.environ["FAL_KEY"] = key
    try:
        out: List[Dict[str, Any]] = []
        for im in input_images:
            if im.get("url"):
                out.append(im)
                continue
            path = DATA_DIR / im["path"]
            if not path.is_file():
                out.append(im)
                continue
            url = fal_client.upload_file(path)
            out.append({**im, "url": url})
        return out
    finally:
        if prev is None:
            os.environ.pop("FAL_KEY", None)
        else:
            os.environ["FAL_KEY"] = prev


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


def build_fal_payload(
    model_id: str,
    prompt: str,
    input_images: List[Dict[str, Any]],
    image_size: Optional[Union[str, Dict[str, int]]],
    aspect_ratio: Optional[str],
    resolution: Optional[str],
    num_images: int,
    guidance_scale: float,
    seed: Optional[int],
    enable_safety_checker: bool,
    safety_tolerance: Optional[str],
    sync_mode: bool,
    limit_generations: Optional[bool],
    enable_web_search: bool,
    output_format: str,
    background: Optional[str] = None,
    quality: Optional[str] = None,
    input_fidelity: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
) -> Dict[str, Any]:
    """Create the fal.ai request payload for a model."""
    model_config = FAL_MODELS.get(model_id, {})
    payload_type = model_config.get("payload_type", "instruct_edit")

    image_urls = []
    for image in input_images:
        if image.get("url"):
            image_urls.append(image["url"])
            continue
        image_path = DATA_DIR / image["path"]
        image_urls.append(build_data_uri(image_path, image["mime_type"]))

    resolved_image_size: Union[str, Dict[str, int]] = "auto"
    if payload_type == "instruct_edit" and model_config.get("image_size_from_first_image") and input_images:
        dims = get_image_dimensions(input_images[0])
        if dims:
            resolved_image_size = {"width": dims[0], "height": dims[1]}
        elif image_size:
            resolved_image_size = image_size if isinstance(image_size, str) else image_size
    elif payload_type == "qwen_pro_edit" and model_config.get("image_size_from_first_image") and input_images:
        dims = get_image_dimensions(input_images[0])
        if dims:
            resolved_image_size = {"width": dims[0], "height": dims[1]}
        elif image_size:
            resolved_image_size = image_size if isinstance(image_size, str) else image_size
    elif image_size:
        resolved_image_size = image_size if isinstance(image_size, str) else image_size

    if payload_type == "qwen_pro_edit":
        payload = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": int(num_images),
            "enable_safety_checker": bool(enable_safety_checker),
            "output_format": output_format,
        }
        if isinstance(resolved_image_size, dict):
            payload["image_size"] = resolved_image_size
        elif isinstance(resolved_image_size, str) and resolved_image_size != "auto":
            payload["image_size"] = resolved_image_size
        if seed is not None:
            payload["seed"] = int(seed)
        return payload

    if payload_type == "hy_wu_edit":
        size_val: Union[str, Dict[str, int]] = (
            image_size if image_size is not None else model_config.get("image_sizes", ["auto"])[0]
        )
        if isinstance(size_val, dict):
            resolved_size: Union[str, Dict[str, int]] = size_val
        else:
            resolved_size = str(size_val)
        steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(model_config.get("num_inference_steps_default", 30))
        )
        thinking = (
            bool(enable_thinking)
            if enable_thinking is not None
            else bool(model_config.get("enable_thinking_default", True))
        )
        payload = {
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": resolved_size,
            "num_inference_steps": steps,
            "num_images": int(num_images),
            "enable_thinking": thinking,
            "enable_safety_checker": bool(enable_safety_checker),
            "output_format": output_format,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if model_config.get("supports_sync_mode") and sync_mode:
            payload["sync_mode"] = True
        return payload

    if payload_type == "nano_banana_edit":
        payload = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": int(num_images),
        }
        if model_config.get("supports_seed") and seed is not None:
            payload["seed"] = int(seed)
        if model_config.get("supports_aspect_ratio") and aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if model_config.get("supports_resolution") and resolution:
            payload["resolution"] = resolution
        if model_config.get("supports_output_format"):
            payload["output_format"] = output_format
        if model_config.get("supports_safety_tolerance") and safety_tolerance:
            payload["safety_tolerance"] = safety_tolerance
        if model_config.get("supports_sync_mode") and sync_mode:
            payload["sync_mode"] = True
        if model_config.get("supports_limit_generations") and limit_generations is not None:
            payload["limit_generations"] = bool(limit_generations)
        if model_config.get("supports_enable_web_search") and enable_web_search:
            payload["enable_web_search"] = True
        return payload

    if payload_type == "gpt_image_edit":
        payload = {
            "prompt": prompt,
            "image_urls": image_urls,
        }
        if model_config.get("supports_num_images"):
            payload["num_images"] = int(num_images)
        if model_config.get("supports_image_size") and image_size:
            payload["image_size"] = image_size
        if model_config.get("supports_output_format"):
            payload["output_format"] = output_format
        if model_config.get("supports_background") and background:
            payload["background"] = background
        if model_config.get("supports_quality") and quality:
            payload["quality"] = quality
        if model_config.get("supports_input_fidelity") and input_fidelity:
            payload["input_fidelity"] = input_fidelity
        if model_config.get("supports_sync_mode") and sync_mode:
            payload["sync_mode"] = True
        return payload

    if payload_type == "gemini_flash_edit_multi":
        return {
            "prompt": prompt,
            "input_image_urls": image_urls,
        }

    if payload_type == "grok_imagine_edit":
        res_raw = (resolution or "1k").strip().lower()
        res = res_raw if res_raw in ("1k", "2k") else "1k"
        payload = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": int(num_images),
            "resolution": res,
            "output_format": output_format,
        }
        if model_config.get("supports_sync_mode") and sync_mode:
            payload["sync_mode"] = True
        return payload

    payload = {
        "prompt": prompt,
        "image_urls": image_urls,
        "image_size": resolved_image_size,
    }
    if model_config.get("supports_num_images"):
        payload["num_images"] = int(num_images)
    if model_config.get("supports_guidance_scale"):
        payload["guidance_scale"] = float(guidance_scale)
    if model_config.get("supports_safety_checker"):
        payload["enable_safety_checker"] = bool(enable_safety_checker)
    if model_config.get("supports_output_format"):
        payload["output_format"] = output_format
    if model_config.get("supports_safety_tolerance") and safety_tolerance:
        payload["safety_tolerance"] = safety_tolerance
    if model_config.get("supports_sync_mode") and sync_mode:
        payload["sync_mode"] = True
    if model_config.get("supports_seed") and seed is not None:
        payload["seed"] = int(seed)
    return payload


def run_fal_request(
    model_id: str,
    prompt: str,
    input_images: List[Dict[str, Any]],
    image_size: Optional[Union[str, Dict[str, int]]],
    aspect_ratio: Optional[str],
    resolution: Optional[str],
    num_images: int,
    guidance_scale: float,
    seed: Optional[int],
    enable_safety_checker: bool,
    safety_tolerance: Optional[str],
    sync_mode: bool,
    limit_generations: Optional[bool],
    enable_web_search: bool,
    output_format: str,
    job_dir: Path,
    background: Optional[str] = None,
    quality: Optional[str] = None,
    input_fidelity: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Run fal.ai request and store returned images."""
    fal_key = os.environ.get("FAL_KEY", "")
    if not fal_key:
        raise RuntimeError("FAL_KEY environment variable is required for fal.ai requests.")

    payload = build_fal_payload(
        model_id=model_id,
        prompt=prompt,
        input_images=input_images,
        image_size=image_size,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        num_images=num_images,
        guidance_scale=guidance_scale,
        seed=seed,
        enable_safety_checker=enable_safety_checker,
        safety_tolerance=safety_tolerance,
        sync_mode=sync_mode,
        limit_generations=limit_generations,
        enable_web_search=enable_web_search,
        output_format=output_format,
        background=background,
        quality=quality,
        input_fidelity=input_fidelity,
        num_inference_steps=num_inference_steps,
        enable_thinking=enable_thinking,
    )

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
    params_path = outputs_dir / "params.json"
    params_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    result_payload = response_data.get("data", response_data)
    output_text = result_payload.get("description") or result_payload.get("revised_prompt", "")
    output_images: List[Dict[str, Any]] = []
    images_payload = result_payload.get("images")
    if images_payload is None and isinstance(result_payload.get("image"), dict):
        images_payload = [result_payload["image"]]
    for idx, image in enumerate(images_payload or [], start=1):
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

    return output_text, output_images, debug_path
