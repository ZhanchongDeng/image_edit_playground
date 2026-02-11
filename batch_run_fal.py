import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fal_client
import requests


MODEL_ID = "fal-ai/qwen-image-edit-2511-multiple-angles"
DEFAULT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def build_data_uri(path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def iter_images(input_dir: Path, extensions: Iterable[str]) -> List[Path]:
    allowed = {ext.lower() for ext in extensions}
    return sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in allowed
        ]
    )


def normalize_fal_result(result: Any) -> Dict[str, Any]:
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


def build_payload(args: argparse.Namespace, image_path: Path) -> Dict[str, Any]:
    mime = guess_mime(image_path)
    payload: Dict[str, Any] = {
        "image_urls": [build_data_uri(image_path, mime)],
    }

    if args.vertical_angle is not None:
        payload["vertical_angle"] = float(args.vertical_angle)
    if args.zoom is not None:
        payload["zoom"] = float(args.zoom)
    if args.additional_prompt:
        payload["additional_prompt"] = args.additional_prompt
    if args.lora_scale is not None:
        payload["lora_scale"] = float(args.lora_scale)
    if args.image_size:
        if args.image_size.strip().startswith("{"):
            payload["image_size"] = json.loads(args.image_size)
        else:
            payload["image_size"] = args.image_size
    if args.guidance_scale is not None:
        payload["guidance_scale"] = float(args.guidance_scale)
    if args.num_inference_steps is not None:
        payload["num_inference_steps"] = int(args.num_inference_steps)
    if args.acceleration:
        payload["acceleration"] = args.acceleration
    if args.negative_prompt is not None:
        payload["negative_prompt"] = args.negative_prompt
    if args.seed is not None:
        payload["seed"] = int(args.seed)
    if args.sync_mode:
        payload["sync_mode"] = True
    if args.enable_safety_checker is not None:
        payload["enable_safety_checker"] = bool(args.enable_safety_checker)
    if args.output_format:
        payload["output_format"] = args.output_format
    if args.num_images is not None:
        payload["num_images"] = int(args.num_images)

    return payload


def create_output_dir(base_dir: Path, stem: str) -> Path:
    target = base_dir / stem
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        return target
    for idx in range(1, 1000):
        candidate = base_dir / f"{stem}_{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    raise RuntimeError(f"Unable to find free output directory for {stem}.")


def build_output_filename(input_path: Path, index: int, ext: str) -> str:
    stem = input_path.stem
    if stem.endswith("_start"):
        stem = f"{stem[:-6]}_end"
    elif stem.endswith("-start"):
        stem = f"{stem[:-6]}-end"
    else:
        stem = f"{stem}_end"
    if index > 1:
        stem = f"{stem}_{index}"
    return f"{stem}.{ext}"


def format_angle(angle: float) -> str:
    if float(angle).is_integer():
        return str(int(angle))
    return str(angle)


def download_outputs(
    result_payload: Dict[str, Any],
    output_dir: Path,
    input_path: Path,
    horizontal_angle: Optional[float] = None,
) -> List[Path]:
    images_payload = result_payload.get("images")
    if images_payload is None and isinstance(result_payload.get("image"), dict):
        images_payload = [result_payload["image"]]

    saved: List[Path] = []
    for idx, image in enumerate(images_payload or [], start=1):
        url = image.get("url")
        if not url:
            continue
        mime = image.get("content_type") or "image/png"
        ext = mime.split("/")[-1] or "png"
        if horizontal_angle is None:
            filename = build_output_filename(input_path, idx, ext)
        else:
            angle_label = format_angle(horizontal_angle)
            if len(images_payload or []) == 1:
                filename = f"{angle_label}.{ext}"
            else:
                filename = f"{angle_label}_{idx}.{ext}"
        target_path = output_dir / filename
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        target_path.write_bytes(response.content)
        saved.append(target_path)
    return saved


def run_for_image(
    args: argparse.Namespace,
    image_path: Path,
    output_dir: Path,
    horizontal_angle: float,
) -> List[Path]:

    def on_queue_update(update: Any) -> None:
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(log.get("message", ""))

    payload = build_payload(args, image_path)
    payload["horizontal_angle"] = float(horizontal_angle)
    result = fal_client.subscribe(
        MODEL_ID,
        arguments=payload,
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    response_data = normalize_fal_result(result)
    result_payload = response_data.get("data", response_data)
    return download_outputs(
        result_payload, output_dir, image_path, horizontal_angle=horizontal_angle
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen Image Edit 2511 Multiple Angles on a folder of images."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing input images.")
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=sorted(DEFAULT_EXTENSIONS),
        help="Allowed input file extensions (default: png jpg jpeg webp).",
    )
    parser.add_argument("--vertical-angle", type=float, help="Vertical angle in degrees.")
    parser.add_argument("--zoom", type=float, help="Zoom level (0-10).")
    parser.add_argument(
        "--additional-prompt",
        default="",
        help="Additional prompt text appended to the generated prompt.",
    )
    parser.add_argument("--lora-scale", type=float, help="LoRA scale factor.")
    parser.add_argument("--image-size", help="Image size enum or JSON width/height.")
    parser.add_argument("--guidance-scale", type=float, help="CFG guidance scale.")
    parser.add_argument("--num-inference-steps", type=int, help="Inference steps.")
    parser.add_argument(
        "--acceleration",
        choices=["none", "regular"],
        help="Acceleration mode.",
    )
    parser.add_argument("--negative-prompt", help="Negative prompt.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--max-input-images",
        type=int,
        help="Maximum number of input images to process from the folder.",
    )
    parser.add_argument(
        "--sync-mode",
        action="store_true",
        help="Return images as data URIs instead of URLs.",
    )
    parser.add_argument(
        "--enable-safety-checker",
        type=int,
        choices=[0, 1],
        help="Enable safety checker (1) or disable (0).",
    )
    parser.add_argument(
        "--output-format",
        choices=["png", "jpeg", "webp"],
        help="Output image format.",
    )
    parser.add_argument("--num-images", type=int, help="Number of images to generate.")
    return parser.parse_args()


def main() -> None:
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY environment variable is required.")

    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    outputs_root = Path(__file__).resolve().parent / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    run_root = create_output_dir(
        outputs_root, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    images = iter_images(input_dir, args.extensions)
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    total_images = len(images)
    if args.max_input_images is not None and args.max_input_images > 0:
        images = images[: args.max_input_images]
    if len(images) < total_images:
        print(
            f"Found {total_images} images in {input_dir}; "
            f"processing first {len(images)}."
        )
    else:
        print(f"Found {len(images)} images in {input_dir}")
    horizontal_angles = list(range(45, 360, 45))
    max_workers = min(8, len(images) * len(horizontal_angles))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: Dict[concurrent.futures.Future[List[Path]], Tuple[Path, float]] = {}
        total_tasks = len(images) * len(horizontal_angles)
        task_idx = 1
        for image_path in images:
            image_output_dir = create_output_dir(run_root, image_path.stem)
            for angle in horizontal_angles:
                print(
                    f"[{task_idx}/{total_tasks}] Queued {image_path.name} at {angle}"
                )
                future = executor.submit(
                    run_for_image, args, image_path, image_output_dir, angle
                )
                futures[future] = (image_path, angle)
                task_idx += 1

        for future in concurrent.futures.as_completed(futures):
            image_path, angle = futures[future]
            try:
                saved_paths = future.result()
            except Exception as exc:
                print(f"Failed {image_path.name} at {angle}: {exc}")
                continue
            for saved_path in saved_paths:
                print(f"Saved {saved_path}")


if __name__ == "__main__":
    main()
