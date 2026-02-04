from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
DEFAULT_CREDENTIALS_PATH = BASE_DIR / "google_cloud.json"
EXAMPLES_DIR = BASE_DIR / "example_inputs"

FAL_MODELS = {
    "fal-ai/hunyuan-image/v3/instruct/edit": {
        "label": "Hunyuan Image Edit",
        "image_sizes": [
            "auto",
            "square_hd",
            "square",
            "portrait_4_3",
            "portrait_16_9",
            "landscape_4_3",
            "landscape_16_9",
        ],
        "output_formats": ["png", "jpeg"],
        "guidance_default": 3.5,
        "max_input_images": 2,
    },
    "fal-ai/qwen-image-edit-2511": {
        "label": "Qwen Image Edit 2511",
        "image_sizes": [
            "square_hd",
            "square",
            "portrait_4_3",
            "portrait_16_9",
            "landscape_4_3",
            "landscape_16_9",
        ],
        "output_formats": ["png", "jpeg", "webp"],
        "guidance_default": 4.5,
        "max_input_images": None,
    },
}
