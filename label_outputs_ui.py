from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import streamlit as st


OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ROTATION_VALUES = [0, 45, 90, 135, 180, 225, 270, 315]
ROTATION_OPTIONS = ["Unlabeled", *[str(value) for value in ROTATION_VALUES]]


def list_runs() -> list[str]:
    if not OUTPUTS_DIR.exists():
        return []
    return sorted([p.name for p in OUTPUTS_DIR.iterdir() if p.is_dir()])


def list_subfolders(run_dir: Path) -> list[str]:
    return sorted([p.name for p in run_dir.iterdir() if p.is_dir()])


def list_images(folder_dir: Path) -> list[Path]:
    return sorted([p for p in folder_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def infer_rotation_from_name(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    try:
        value = int(stem)
    except ValueError:
        return None
    return value if value in ROTATION_VALUES else None


def load_labels(run_dir: Path) -> Dict[str, Dict[str, Optional[int]]]:
    labels_path = run_dir / "labels.json"
    if not labels_path.exists():
        return {}
    try:
        with labels_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_labels(run_dir: Path, data: Dict[str, Dict[str, Optional[int]]]) -> None:
    labels_path = run_dir / "labels.json"
    with labels_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def next_available_name(
    folder_dir: Path,
    desired_name: str,
    existing_names: set[str],
) -> str:
    if desired_name not in existing_names and not (folder_dir / desired_name).exists():
        return desired_name
    stem = Path(desired_name).stem
    suffix = Path(desired_name).suffix
    counter = 1
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in existing_names and not (folder_dir / candidate).exists():
            return candidate
        counter += 1


def cleanup_rotation_suffixes(folder_dir: Path) -> int:
    renamed = 0
    for image_path in list_images(folder_dir):
        stem = image_path.stem
        suffix = image_path.suffix
        if not stem.endswith("_1"):
            continue
        base_stem = stem[:-2]
        try:
            base_value = int(base_stem)
        except ValueError:
            continue
        if base_value not in ROTATION_VALUES:
            continue
        target = folder_dir / f"{base_stem}{suffix}"
        if not target.exists():
            image_path.rename(target)
            renamed += 1
    return renamed


st.set_page_config(page_title="Output Labeling", layout="wide")
st.title("Output Labeling")

if not OUTPUTS_DIR.exists():
    st.error(f"Outputs directory not found: {OUTPUTS_DIR}")
    st.stop()

runs = list_runs()
if not runs:
    st.info("No runs found in outputs.")
    st.stop()

run_name = st.selectbox("Run", runs, index=0)
run_dir = OUTPUTS_DIR / run_name
labels_data = load_labels(run_dir)

subfolders = list_subfolders(run_dir)
if not subfolders:
    st.info("No subfolders found in the selected run.")
    st.stop()

if st.session_state.get("active_run") != run_name:
    st.session_state["active_run"] = run_name
    st.session_state["subfolder_select"] = subfolders[0]

if st.session_state.get("subfolder_select") not in subfolders:
    st.session_state["subfolder_select"] = subfolders[0]

col_prev, col_mid, col_next = st.columns([1, 2, 1])
with col_prev:
    if st.button("◀ Prev", use_container_width=True):
        current_idx = subfolders.index(st.session_state["subfolder_select"])
        if current_idx > 0:
            st.session_state["subfolder_select"] = subfolders[current_idx - 1]
            st.rerun()
with col_next:
    if st.button("Next ▶", use_container_width=True):
        current_idx = subfolders.index(st.session_state["subfolder_select"])
        if current_idx < len(subfolders) - 1:
            st.session_state["subfolder_select"] = subfolders[current_idx + 1]
            st.rerun()

subfolder = st.selectbox("Subfolder", subfolders, key="subfolder_select")
folder_dir = run_dir / subfolder

images = list_images(folder_dir)
if not images:
    st.info("No images found in this subfolder.")
    st.stop()

st.caption(f"{len(images)} image(s) in {subfolder}")

cols = st.columns(3)
for idx, image_path in enumerate(images):
    col = cols[idx % len(cols)]
    with col:
        st.image(str(image_path), use_container_width=True)
        st.caption(image_path.name)

        existing_rotation = labels_data.get(subfolder, {}).get(image_path.name)
        default_rotation = existing_rotation if existing_rotation is not None else infer_rotation_from_name(image_path.name)
        rotation_key = f"rotation::{subfolder}::{image_path.name}"
        delete_key = f"delete::{subfolder}::{image_path.name}"

        if rotation_key not in st.session_state:
            st.session_state[rotation_key] = (
                str(default_rotation) if default_rotation in ROTATION_VALUES else "Unlabeled"
            )
        if delete_key not in st.session_state:
            st.session_state[delete_key] = False

        st.selectbox("Rotation", ROTATION_OPTIONS, key=rotation_key)
        st.checkbox("Delete", key=delete_key)

if st.button("Save labels", type="primary", use_container_width=True):
    updated = labels_data.copy()
    updated[subfolder] = {}
    to_delete: list[Path] = []
    to_keep: list[Path] = []
    existing_names = {image_path.name for image_path in images}

    for image_path in images:
        delete_key = f"delete::{subfolder}::{image_path.name}"
        if st.session_state.get(delete_key):
            to_delete.append(image_path)
        else:
            to_keep.append(image_path)

    deleted_count = 0
    for image_path in to_delete:
        if image_path.exists():
            image_path.unlink()
            deleted_count += 1
        existing_names.discard(image_path.name)

    renamed_count = 0
    for image_path in to_keep:
        rotation_key = f"rotation::{subfolder}::{image_path.name}"
        rotation_value = st.session_state.get(rotation_key, "Unlabeled")
        rotation = int(rotation_value) if rotation_value != "Unlabeled" else None

        target_name = image_path.name
        if rotation is not None:
            desired_name = f"{rotation}{image_path.suffix}"
            if desired_name != image_path.name:
                target_name = next_available_name(folder_dir, desired_name, existing_names)
                existing_names.discard(image_path.name)
                existing_names.add(target_name)
                image_path.rename(folder_dir / target_name)
                renamed_count += 1

        updated[subfolder][target_name] = rotation

    cleanup_count = cleanup_rotation_suffixes(folder_dir)

    save_labels(run_dir, updated)

    st.success(
        f"Saved labels for {subfolder}. Renamed {renamed_count} image(s). "
        f"Deleted {deleted_count} image(s). Cleaned {cleanup_count} collision suffixes."
    )
    st.rerun()
