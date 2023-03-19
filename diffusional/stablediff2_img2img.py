import torch

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


def init_pipeline(is_img2img=False):
    if is_img2img:
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
        )
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
        )
    pipeline = pipeline.to("cuda")
    pipeline.enable_attention_slicing()
    return pipeline


def generate_images(pipeline, prompt: str, img: str = None, n: int = 1):
    # TODO image name and image type
    print("Generating static with Stable Diffusion v2.1 for the following prompt:")
    print(f'  "{prompt}"')
    if img:
        print(f"  (based on image: {img})")

    if img:
        with Image.open("/home/markus/Pictures/forrest.png") as init_image:
            init_image = init_image.convert("RGB")
            init_image = init_image.resize((786, 512))
            images = pipeline(
                prompt=prompt, image=init_image, strength=1, num_images_per_prompt=n
            ).images
    else:
        images = pipeline(prompt, num_images_per_prompt=n).images

    for i in range(n):
        image = images[i]
        image_file = f"image-{i}.png"
        image.save(image_file)
