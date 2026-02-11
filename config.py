from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
DEFAULT_CREDENTIALS_PATH = BASE_DIR / "google_cloud.json"
EXAMPLES_DIR = BASE_DIR / "example_inputs"

VERTEX_MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image-preview",
]

FAL_MODELS = {
    "fal-ai/hunyuan-image/v3/instruct/edit": {
        "label": "Hunyuan Image Edit",
        "payload_type": "instruct_edit",
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
        "supports_num_images": True,
        "supports_guidance_scale": True,
        "supports_seed": True,
        "supports_safety_checker": True,
        "supports_output_format": True,
        "supports_image_size": True,
    },
    "fal-ai/qwen-image-edit-2511": {
        "label": "Qwen Image Edit 2511",
        "payload_type": "instruct_edit",
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
        "supports_num_images": True,
        "supports_guidance_scale": True,
        "supports_seed": True,
        "supports_safety_checker": True,
        "supports_output_format": True,
        "supports_image_size": True,
    },
    "fal-ai/nano-banana/edit": {
        "label": "Nano Banana Edit",
        "payload_type": "nano_banana_edit",
        "image_sizes": ["auto"],
        "output_formats": ["png", "jpeg", "webp"],
        "guidance_default": 0.0,
        "max_input_images": None,
        "supports_num_images": True,
        "supports_guidance_scale": False,
        "supports_seed": False,
        "supports_safety_checker": False,
        "supports_output_format": True,
        "supports_image_size": False,
    },
    "fal-ai/gemini-flash-edit/multi": {
        "label": "Gemini Flash Edit Multi",
        "payload_type": "gemini_flash_edit_multi",
        "image_sizes": ["auto"],
        "output_formats": ["png"],
        "guidance_default": 0.0,
        "max_input_images": None,
        "supports_num_images": False,
        "supports_guidance_scale": False,
        "supports_seed": False,
        "supports_safety_checker": False,
        "supports_output_format": False,
        "supports_image_size": False,
    },
}
