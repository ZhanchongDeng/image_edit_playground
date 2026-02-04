import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import streamlit as st
from google.genai import types

from config import DATA_DIR, DEFAULT_CREDENTIALS_PATH, FAL_MODELS, JOBS_DIR
from examples import apply_example_input, load_example_inputs
from fal_provider import parse_optional_int, run_fal_instruct_edit
from io_utils import save_example_images, save_uploaded_images
from storage import append_job, load_history, update_job
from vertex_provider import (
    build_config,
    build_parts,
    create_client,
    parse_response,
    save_response_debug,
)


def apply_history_job(job: Dict[str, Any]) -> None:
    """Stage a history job to reapply its prompt."""
    st.session_state["pending_apply"] = {
        "prompt": job.get("prompt", ""),
    }


st.set_page_config(page_title="Image Editing Playground", layout="wide")
st.title("Image Editing Playground")

# Sidebar: connection and model configuration.
with st.sidebar:
    st.header("Connection")
    st.header("Model & Settings")
    model_choice = st.selectbox(
        "Model",
        ["gemini-3-pro-image-preview", *FAL_MODELS.keys()],
        index=0,
    )

    if model_choice == "gemini-3-pro-image-preview":
        st.markdown("Vertex AI requires OAuth2/ADC credentials (API keys are not supported).")
        credentials_path = st.text_input(
            "GOOGLE_APPLICATION_CREDENTIALS (optional)",
            value=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            or (str(DEFAULT_CREDENTIALS_PATH) if DEFAULT_CREDENTIALS_PATH.exists() else ""),
            type="password",
        )
        project = st.text_input("GCP Project (optional)", value=os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
        location = st.text_input("GCP Location (optional)", value=os.environ.get("GOOGLE_CLOUD_LOCATION", ""))
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
    else:
        st.markdown("fal.ai requires `FAL_KEY` to be set in the environment.")
        fal_config = FAL_MODELS[model_choice]
        fal_key_input = st.text_input(
            "FAL_KEY (optional)",
            value=os.environ.get("FAL_KEY", ""),
            type="password",
        )
        fal_image_size = st.selectbox(
            "Image size",
            fal_config["image_sizes"],
            index=0,
        )
        fal_num_images = st.number_input("Num images", min_value=1, max_value=4, value=1)
        fal_guidance_scale = st.slider(
            "Guidance scale",
            min_value=0.0,
            max_value=20.0,
            value=float(fal_config["guidance_default"]),
            step=0.1,
        )
        fal_seed_text = st.text_input("Seed (optional)", value="")
        fal_enable_safety = st.checkbox("Enable safety checker", value=True)
        fal_output_format = st.selectbox("Output format", fal_config["output_formats"], index=0)

# Submit form and inputs.
st.subheader("Submit job")
examples = load_example_inputs()
if examples:
    st.write("Example inputs")
    with st.expander("Show example inputs", expanded=False):
        example_labels = ["(none)", *[example["label"] for example in examples]]
        selected_label = st.selectbox("Choose example", example_labels, index=0)
        selected_example = next((ex for ex in examples if ex["label"] == selected_label), None)
        if selected_example:
            if selected_example.get("prompt"):
                st.write("Example prompt:")
                st.code(selected_example["prompt"])
            if selected_example.get("images"):
                st.write("Example images:")
                cols = st.columns(min(4, len(selected_example["images"])))
                for idx, image in enumerate(selected_example["images"]):
                    with cols[idx % len(cols)]:
                        st.image(image["source_path"], caption=image["filename"])
            st.button(
                "Use this example",
                key=f"use_example_{selected_example['id']}",
                on_click=apply_example_input,
                args=(selected_example,),
            )
        if st.button("Clear example selection"):
            st.session_state["example_images"] = []
            st.session_state["example_label"] = ""

# Apply a prompt from history if the user clicked "Apply prompt".
pending_apply = st.session_state.pop("pending_apply", None)
if pending_apply:
    st.session_state["prompt_text"] = pending_apply.get("prompt", "")
prompt = st.text_area(
    "Prompt",
    height=140,
    placeholder="Describe the edit you want...",
    key="prompt_text",
)
uploads = st.file_uploader(
    "Input images",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)
example_images = st.session_state.get("example_images", [])
example_label = st.session_state.get("example_label", "")
if example_images:
    if uploads:
        st.caption("Uploads are being used (example images are ignored).")
    else:
        st.caption(f"Using example images: {example_label}")
        cols = st.columns(min(4, len(example_images)))
        for idx, image in enumerate(example_images):
            with cols[idx % len(cols)]:
                st.image(image["source_path"], caption=image["filename"])
col_submit, col_retry = st.columns(2)
with col_submit:
    submit = st.button("Submit job", type="primary", use_container_width=True)
with col_retry:
    submit_with_retry = st.button("Submit + retry up to 5 min", use_container_width=True)

if submit or submit_with_retry:
    # Validate inputs and build the job payload.
    can_submit = True
    seed_value = None
    input_count = len(uploads) if uploads else len(example_images)
    if model_choice in FAL_MODELS:
        if not prompt.strip():
            st.error("fal.ai requires a prompt.")
            can_submit = False
        elif input_count == 0:
            st.error("fal.ai requires at least one input image.")
            can_submit = False
        else:
            max_images = FAL_MODELS[model_choice]["max_input_images"]
            if max_images and input_count > max_images:
                st.error(f"fal.ai supports a maximum of {max_images} input images for this model.")
                can_submit = False
            try:
                seed_value = parse_optional_int(fal_seed_text)
            except ValueError as exc:
                st.error(str(exc))
                can_submit = False
    elif not prompt and not uploads:
        st.error("Add a prompt or at least one image.")
        can_submit = False

    if can_submit:
        # Persist job metadata and prepare inputs on disk.
        job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        input_images = save_uploaded_images(job_dir, uploads or [])
        if not input_images and example_images:
            input_images = save_example_images(job_dir, example_images)
        job_payload = {
            "id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "queued",
            "prompt": prompt,
            "model": model_choice,
            "provider": "vertex" if model_choice == "gemini-3-pro-image-preview" else "fal",
            "settings": {},
            "input_images": input_images,
            "outputs": {"text": "", "images": []},
            "error": "",
        }
        if model_choice == "gemini-3-pro-image-preview":
            job_payload["settings"] = {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
                "response_modalities": response_modalities,
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
                "output_mime_type": output_mime_type,
            }
        else:
            job_payload["settings"] = {
                "model_id": model_choice,
                "image_size": fal_image_size,
                "num_images": int(fal_num_images),
                "guidance_scale": float(fal_guidance_scale),
                "seed": seed_value,
                "enable_safety_checker": fal_enable_safety,
                "output_format": fal_output_format,
            }
        append_job(job_payload)
        update_job(job_id, {"status": "running"})

        # Execute provider request with optional retries.
        spinner_label = "Running request on Vertex AI..." if model_choice == "gemini-3-pro-image-preview" else "Running request on fal.ai..."
        with st.spinner(spinner_label):
            attempts = 0
            deadline = time.time() + (5 * 60 if submit_with_retry else 0)
            last_error = ""
            while True:
                attempts += 1
                try:
                    if model_choice == "gemini-3-pro-image-preview":
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
                            model=model_choice,
                            contents=contents,
                            config=config,
                        )
                        debug_path = save_response_debug(response, job_dir)
                        output_text, output_images = parse_response(response, job_dir)
                    else:
                        if fal_key_input:
                            os.environ["FAL_KEY"] = fal_key_input
                        output_text, output_images, debug_path = run_fal_instruct_edit(
                            model_id=model_choice,
                            prompt=prompt,
                            input_images=input_images,
                            image_size=fal_image_size,
                            num_images=int(fal_num_images),
                            guidance_scale=float(fal_guidance_scale),
                            seed=seed_value,
                            enable_safety_checker=fal_enable_safety,
                            output_format=fal_output_format,
                            job_dir=job_dir,
                        )
                    if submit_with_retry and not output_images:
                        # Retry when no images are returned within the window.
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

# Job history display.
st.subheader("Job history")
history = load_history()

status_filter = st.selectbox("Status filter", ["all", "success", "failed"], index=0)

filtered_history = [job for job in history if job.get("model") == model_choice]
if status_filter == "success":
    filtered_history = [job for job in filtered_history if job.get("status") == "completed"]
elif status_filter == "failed":
    filtered_history = [job for job in filtered_history if job.get("status") == "failed"]

if not filtered_history:
    st.info("No jobs submitted yet.")
else:
    for job in filtered_history:
        header = f"{job['id']} • {job.get('status', 'unknown')}"
        with st.expander(header, expanded=False):
            st.button(
                "Apply prompt",
                key=f"apply_{job['id']}",
                on_click=apply_history_job,
                args=(job,),
            )
            if job.get("status") == "completed":
                st.success("Job succeeded.")
            elif job.get("status") == "failed":
                st.error("Job failed.")
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

