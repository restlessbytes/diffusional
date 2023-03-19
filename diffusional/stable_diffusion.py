import torch

from pathlib import Path
from diffusers import StableDiffusionPipeline

from diffusional.config import APPCONFIG

STATIC_DIR = "diffusional/static"


"""
Module with functions for image generation using HuggingFace's Stable Diffusion Pipeline:

https://huggingface.co/docs/diffusers/v0.3.0/en/api/pipelines/stable_diffusion
"""


class ImageGenerationParameters:
    """
    A class that represents input parameters for the image generation process.

    It's basically a data class with some additional input validation.
    """

    def __init__(self, form_data: dict):
        """Reads image generation parameters from some HTML form data.

        Expects the following parameters:
            prompt -- The prompt that'll be used to guide the image generation process
            num -- Number of images to be generated
            guide -- Guidance scale. Higher values generate images more closely linked to the prompt at the expense of image quality
            interfere -- Number of interference (or, better: denoising) steps. Higher values improve image quality

        For more explanation on those params, see: https://huggingface.co/docs/diffusers/v0.3.0/en/api/pipelines/stable_diffusion

        :param form_data: A dict containing input parameters as described above.
        """
        prompt: str = form_data["prompt"]
        num_of_images: int = int(form_data.get("num", "1").strip())
        guidance_scale: float = float(form_data.get("guide", "0.75").strip())
        interference_steps: int = int(form_data.get("interfere", "15").strip())
        assert prompt, "No prompt provided for image generation!"
        assert (
            1 <= num_of_images <= 5
        ), "Only 1 - 5 images are allowed to be generated per run!"
        assert (
            0.0 <= guidance_scale <= 1.0
        ), "Guidance scale value must be between 0 and 1!"
        assert (
            5 <= interference_steps <= 100
        ), "No. of interference steps must be between 5 and 100!"
        self.prompt: str = prompt
        self.num_of_images: int = num_of_images
        self.guidance_scale: float = guidance_scale
        self.interference_steps: int = interference_steps


def init_pipeline(model_config: dict) -> StableDiffusionPipeline:
    """
    Initialize a `StableDiffusionPipeline`: https://huggingface.co/docs/diffusers/v0.3.0/en/api/pipelines/stable_diffusion

    The model used for the pipeline is defined either by model_id (which will then fetch the model from huggingface.co)
    or by model_path (which assumes that that's a path that points to a snapshot of that model on disk)

    If both model_id and model_path are define, model_path takes precedence over model_id.

    :param model_config: A simple dictionary used to define the pipeline. See `config.py` as an example.
    :return: A stable diffusion pipeline object.
    """
    if model_config.get("model_path"):
        model = model_config.get("model_path")
    elif model_config.get("model_id"):
        model = model_config.get("model_id")
    else:
        model = None
    if model_config.get("allow_nsfw", False):
        pipe = StableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
    return pipe.to("cuda")


def save_generated_images(prefix: str) -> tuple[str, list[str]]:
    """
    'Saving' images here means: Create a folder named <prefix> and move all files from folder tmp/ to there.

    :param prefix: Name of the folder where the images will be moved to. Will be created if it does not exist.
    :return: The prompt that was used to generate those images + the names of the image files.
    """
    image_folder = Path(STATIC_DIR, prefix)
    image_folder.mkdir(exist_ok=True)
    images = []
    for file in Path(STATIC_DIR, "tmp").iterdir():
        filename = file.stem + file.suffix
        file.rename(Path(STATIC_DIR, prefix, filename))
        if file.suffix == ".png":
            images.append(str(Path(prefix, filename)))
    with open(Path(image_folder, "prompt.txt"), "r") as file:
        prompt = file.read()
    return prompt, images


def generate_images(
    image_gen_params: ImageGenerationParameters, pipe=None
) -> list[str]:
    """
    Generate images according to the parameters defined in image_gen_params.

    Generated images can be found in static/tmp/

    :param image_gen_params: Parameters used to guide and tweak image generation
    :param pipe: An instance of a HuggingFace Stable Diffusion Pipeline
    :return: List of file names that point to the images that were generated
    """
    prompt = image_gen_params.prompt
    if pipe is None:
        pipe = init_pipeline(APPCONFIG["model"])
    print(f"prompt={prompt}")
    prompts = [prompt] * image_gen_params.num_of_images
    images = pipe(
        prompts,
        num_inference_steps=image_gen_params.interference_steps,
        guidance_scale=image_gen_params.guidance_scale,
    ).images
    files = []
    for i, image in enumerate(images):
        filename = f"tmp/image-{i}.png"
        image.save(Path(STATIC_DIR, filename))
        files.append(filename)
    with open(Path(STATIC_DIR, "tmp", "prompt.txt"), "w+") as file:
        file.write(prompt)
    return files
