from pathlib import Path

APPCONFIG = {
    "model": {
        "name": "your-model-name",
        "model_id": "your-model-id",
        "model_path": Path("path/to/model/snapshot"),
        "allow_nsfw": False,
    },
    "defaults": {
        # see: https://huggingface.co/docs/diffusers/v0.3.0/en/api/pipelines/stable_diffusion
        # number of images to be generated per run
        "num": 1,
        # guidance scale
        "guide": 0.75,
        # number of interference steps
        "interfere": 50,
    },
}
