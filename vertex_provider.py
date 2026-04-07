import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from google import genai
from google.genai import types
from google.oauth2 import service_account

from config import DATA_DIR

VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def create_client(
    project: str, location: str, credentials_path: Optional[str] = None
) -> genai.Client:
    """Create a Vertex AI client with explicit service-account credentials."""
    creds_file = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    client_kwargs: Dict[str, Any] = {"vertexai": True}
    if creds_file and Path(creds_file).is_file():
        creds = service_account.Credentials.from_service_account_file(
            creds_file, scopes=VERTEX_SCOPES
        )
        client_kwargs["credentials"] = creds
        if not project:
            project = creds.project_id
    if project:
        client_kwargs["project"] = project
    if location:
        client_kwargs["location"] = location
    return genai.Client(**client_kwargs)


def build_parts(prompt: str, input_images: List[Dict[str, Any]]) -> List[types.Part]:
    """Build Vertex AI content parts from prompt and images (local paths or http(s) URLs)."""
    parts: List[types.Part] = []
    if prompt.strip():
        parts.append(types.Part.from_text(text=prompt))
    for image in input_images:
        if image.get("url"):
            resp = requests.get(image["url"], timeout=60)
            resp.raise_for_status()
            mime = (image.get("mime_type") or "").strip()
            if not mime:
                mime = resp.headers.get("Content-Type", "image/png")
                mime = mime.split(";")[0].strip() or "image/png"
            parts.append(types.Part.from_bytes(data=resp.content, mime_type=mime))
            continue
        image_path = DATA_DIR / image["path"]
        image_bytes = image_path.read_bytes()
        parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=image["mime_type"],
            )
        )
    return parts


def build_config(
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    response_modalities: List[str],
    aspect_ratio: str,
    image_size: str,
    output_mime_type: str,
) -> types.GenerateContentConfig:
    """Create the Vertex AI generation config."""
    return types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_modalities=response_modalities,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            output_mime_type=output_mime_type,
        ),
    )


def parse_response(response: types.GenerateContentResponse, job_dir: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract text and image outputs from a Vertex response."""
    outputs_dir = job_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    text_parts: List[str] = []
    image_outputs: List[Dict[str, Any]] = []

    for candidate in response.candidates or []:
        content = candidate.content
        if not content:
            continue
        for part in content.parts or []:
            if getattr(part, "text", None):
                text_parts.append(part.text)
                continue
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                mime = inline_data.mime_type or "image/png"
                ext = mime.split("/")[-1] or "png"
                filename = f"output_{len(image_outputs) + 1}.{ext}"
                target_path = outputs_dir / filename
                target_path.write_bytes(inline_data.data)
                image_outputs.append(
                    {
                        "path": str(target_path.relative_to(DATA_DIR)),
                        "mime_type": mime,
                    }
                )
                continue
            file_data = getattr(part, "file_data", None)
            if file_data and getattr(file_data, "file_uri", None):
                text_parts.append(f"[file_uri] {file_data.file_uri}")

    return "\n".join(text_parts).strip(), image_outputs


def save_response_debug(response: types.GenerateContentResponse, job_dir: Path) -> str:
    """Persist the raw response for debugging."""
    outputs_dir = job_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    target_path = outputs_dir / "response.json"
    text = ""
    try:
        if hasattr(response, "to_json"):
            text = response.to_json()
        elif hasattr(response, "model_dump_json"):
            text = response.model_dump_json()
        elif hasattr(response, "json"):
            text = response.json()
        else:
            text = repr(response)
    except Exception as exc:  # noqa: BLE001
        text = f"Failed to serialize response: {exc}\n\nrepr:\n{repr(response)}"
    target_path.write_text(text, encoding="utf-8")
    return str(target_path.relative_to(DATA_DIR))
