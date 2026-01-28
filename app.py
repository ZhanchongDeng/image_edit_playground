import mimetypes
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import streamlit as st
from google import genai
from google.genai import types

from storage import append_job, load_history, update_job


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
DEFAULT_CREDENTIALS_PATH = BASE_DIR / "google_cloud.json"


def guess_mime(filename: str, default: str = "application/octet-stream") -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or default


def save_uploaded_images(job_dir: Path, uploads: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Dict[str, Any]]:
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


def build_parts(prompt: str, input_images: List[Dict[str, Any]]) -> List[types.Part]:
    parts: List[types.Part] = []
    if prompt.strip():
        parts.append(types.Part.from_text(text=prompt))
    for image in input_images:
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


def create_client(project: str, location: str) -> genai.Client:
    client_kwargs: Dict[str, Any] = {"vertexai": True}
    if project:
        client_kwargs["project"] = project
    if location:
        client_kwargs["location"] = location
    return genai.Client(**client_kwargs)


st.set_page_config(page_title="NanoBananaPro Playground", layout="wide")
st.title("NanoBananaPro Playground")

with st.sidebar:
    st.header("Connection")
    st.markdown("Vertex AI requires OAuth2/ADC credentials (API keys are not supported).")
    credentials_path = st.text_input(
        "GOOGLE_APPLICATION_CREDENTIALS (optional)",
        value=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        or (str(DEFAULT_CREDENTIALS_PATH) if DEFAULT_CREDENTIALS_PATH.exists() else ""),
        type="password",
    )
    project = st.text_input("GCP Project (optional)", value=os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    location = st.text_input("GCP Location (optional)", value=os.environ.get("GOOGLE_CLOUD_LOCATION", ""))
    st.header("Model & Settings")
    model_name = st.text_input("Model", value="gemini-3-pro-image-preview")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.05)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    max_output_tokens = st.number_input("Max output tokens", min_value=1, max_value=32768, value=32768)
    response_modalities = []
    if st.checkbox("Return text", value=True):
        response_modalities.append("TEXT")
    if st.checkbox("Return images", value=True):
        response_modalities.append("IMAGE")
    aspect_ratio = st.selectbox("Aspect ratio", ["1:1", "4:3", "3:4", "16:9", "9:16"], index=0)
    image_size = st.selectbox("Image size", ["256", "512", "1K", "2K"], index=2)
    output_mime_type = st.selectbox("Output mime type", ["image/png", "image/jpeg", "image/webp"], index=0)

st.subheader("Submit job")
prompt = st.text_area("Prompt", height=140, placeholder="Describe the edit you want...")
uploads = st.file_uploader("Input images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
col_submit, col_retry = st.columns(2)
with col_submit:
    submit = st.button("Submit job", type="primary", use_container_width=True)
with col_retry:
    submit_with_retry = st.button("Submit + retry up to 5 min", use_container_width=True)

if submit or submit_with_retry:
    if not prompt and not uploads:
        st.error("Add a prompt or at least one image.")
    else:
        job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        input_images = save_uploaded_images(job_dir, uploads or [])
        job_payload = {
            "id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "queued",
            "prompt": prompt,
            "model": model_name,
            "settings": {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
                "response_modalities": response_modalities,
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
                "output_mime_type": output_mime_type,
            },
            "input_images": input_images,
            "outputs": {"text": "", "images": []},
            "error": "",
        }
        append_job(job_payload)
        update_job(job_id, {"status": "running"})

        with st.spinner("Running request on Vertex AI..."):
            attempts = 0
            deadline = time.time() + (5 * 60 if submit_with_retry else 0)
            last_error = ""
            while True:
                attempts += 1
                try:
                    if credentials_path:
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
                    client = create_client(project, location)
                    parts = build_parts(prompt, input_images)
                    contents = [types.Content(role="user", parts=parts)]
                    config = build_config(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=int(max_output_tokens),
                        response_modalities=response_modalities,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                        output_mime_type=output_mime_type,
                    )
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    debug_path = save_response_debug(response, job_dir)
                    output_text, output_images = parse_response(response, job_dir)
                    if submit_with_retry and not output_images:
                        if time.time() >= deadline:
                            update_job(
                                job_id,
                                {
                                    "status": "failed",
                                    "error": "No images returned within retry window.",
                                    "attempts": attempts,
                                    "debug": {"response_path": debug_path},
                                },
                            )
                            st.error("Job failed: no images returned within retry window.")
                            break
                        update_job(
                            job_id,
                            {
                                "status": "retrying",
                                "error": "No images returned yet. Retrying...",
                                "attempts": attempts,
                                "debug": {"response_path": debug_path},
                            },
                        )
                        time.sleep(10)
                        continue

                    update_job(
                        job_id,
                        {
                            "status": "completed",
                            "outputs": {
                                "text": output_text,
                                "images": output_images,
                            },
                            "debug": {"response_path": debug_path},
                            "attempts": attempts,
                        },
                    )
                    st.success(f"Job completed (attempt {attempts}).")
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    if not submit_with_retry or time.time() >= deadline:
                        update_job(
                            job_id,
                            {"status": "failed", "error": last_error, "attempts": attempts},
                        )
                        st.error(f"Job failed: {last_error}")
                        break
                    update_job(job_id, {"status": "retrying", "error": last_error, "attempts": attempts})
                    time.sleep(10)

st.subheader("Job history")
history = load_history()

if not history:
    st.info("No jobs submitted yet.")
else:
    for job in history:
        header = f"{job['id']} • {job.get('status', 'unknown')}"
        with st.expander(header, expanded=False):
            st.caption(f"Created: {job.get('created_at', '')}")
            st.write(f"Model: {job.get('model', '')}")
            st.write("Prompt:")
            st.code(job.get("prompt", "") or "(empty)")

            if job.get("input_images"):
                st.write("Input images:")
                cols = st.columns(min(4, len(job["input_images"])))
                for idx, image in enumerate(job["input_images"]):
                    image_path = DATA_DIR / image["path"]
                    with cols[idx % len(cols)]:
                        if image_path.exists():
                            st.image(str(image_path), caption=image["filename"])
                        else:
                            st.text(image["filename"])

            outputs = job.get("outputs", {})
            if outputs.get("text"):
                st.write("Output text:")
                st.code(outputs["text"])

            output_images = outputs.get("images", [])
            if output_images:
                st.write("Output images:")
                cols = st.columns(min(4, len(output_images)))
                for idx, image in enumerate(output_images):
                    image_path = DATA_DIR / image["path"]
                    with cols[idx % len(cols)]:
                        if image_path.exists():
                            st.image(str(image_path))
                        else:
                            st.text(image["path"])

            debug_info = job.get("debug", {})
            debug_path = debug_info.get("response_path")
            if debug_path:
                debug_full_path = DATA_DIR / debug_path
                if debug_full_path.exists():
                    st.write("Response debug:")
                    st.code(str(debug_full_path))

            if job.get("error"):
                st.error(job["error"])

