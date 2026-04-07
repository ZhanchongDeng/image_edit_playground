# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A local Streamlit playground for submitting image editing requests to Google Vertex AI, fal.ai, and BFL Labs partner APIs. Jobs, inputs, and outputs are persisted to disk in `data/`.

## Running the App

```bash
# Main UI
streamlit run app.py

# Output labeling UI (for LoRA/multi-view batch outputs)
streamlit run label_outputs_ui.py

# Batch processing across multiple angles
python batch_run_fal.py /path/to/images --vertical-angle 0 --zoom 1.0

# Job history cleanup utility
python scripts/job_distribution.py [--before DATE] [--keep-days N] [--dry-run]
```

## Environment Variables

```bash
# Vertex AI
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/google_cloud.json"
export GOOGLE_CLOUD_PROJECT="your-project"    # optional
export GOOGLE_CLOUD_LOCATION="us-central1"   # optional

# fal.ai
export FAL_KEY="your-fal-key"

# BFL Labs (partner endpoints such as FLUX.2 Pro fashion eval — project-specific key, not your fal.ai key)
export BFL_API_KEY="your-bfl-project-key"
# Optional: labs hosts may use certs that fail Python’s default verify (official curl examples use -k).
# Default in code is verify off; set this to enable TLS verification:
# export BFL_SSL_VERIFY=1
```

Set these in the same shell you use to run Streamlit (e.g. `~/.zshrc` or `~/.bashrc` on macOS/Linux, or run `export BFL_API_KEY=...` in the terminal before `streamlit run app.py`). You can also paste `BFL_API_KEY` in the app sidebar if you prefer not to persist it in a file.

The default credentials path is `google_cloud.json` in the repo root (gitignored).

## Architecture

**Core modules:**
- `app.py` — Streamlit UI: job submission, history display, settings per model
- `config.py` — All model definitions (Vertex + fal.ai + BFL Labs) with per-model capability flags and defaults; data directory paths
- `vertex_provider.py` — Builds Vertex AI requests and parses responses via `google-genai`
- `fal_provider.py` — Builds fal.ai payloads (model-specific formats), submits requests, downloads outputs
- `bfl_provider.py` — BFL Labs `labs.us2.bfl.ai` partner APIs (submit + poll + download)
- `storage.py` — Load/save/append/update `data/history.json`
- `io_utils.py` — Image file upload, data URI encoding, MIME type detection
- `examples.py` — Loads example inputs from `example_inputs/` directories

**Secondary apps:**
- `label_outputs_ui.py` — Streamlit UI for rotating/labeling batch output images
- `batch_run_fal.py` — Parallel batch runner across 8 horizontal angles using thread pool

**Data layout:**
```
data/
  history.json          # array of job objects, newest first
  jobs/{job_id}/
    inputs/             # user-uploaded images
    outputs/            # generated images + response.json
example_inputs/         # gitignored; example {prompt.txt + images} directories
```

## Job Object Shape

```python
{
    "id": "20260305T183634Z_a1b2c3d4",
    "status": "completed|failed|running|queued|retrying",
    "prompt": "...",
    "model": "model_id",
    "provider": "vertex|fal|bfl",
    "settings": {/* model-specific */},
    "input_images": [{"filename": ..., "path": ..., "mime_type": ..., "url": ...}],
    "outputs": {"text": "...", "images": [{"path": ..., "source_url": ...}]},
    "error": "...",
    "attempts": 1
}
```

## Adding a New fal.ai Model

All model configuration lives in `config.py` in the `FAL_MODELS` dict. Each entry specifies:
- `payload_type` — which payload builder branch in `fal_provider.py` to use
- `supports_*` flags — controls which settings appear in the UI
- `*_default` values — pre-filled UI defaults
- `image_sizes`, `aspect_ratios`, `resolutions` — valid options for dropdowns

After adding to `config.py`, the model appears automatically in the UI and history display. No changes to `app.py` are typically needed unless the model requires a new payload type.

## Adding a New Vertex AI Model

Add the model ID string to `VERTEX_MODELS` in `config.py`. Vertex models share a single request/response format in `vertex_provider.py`.
