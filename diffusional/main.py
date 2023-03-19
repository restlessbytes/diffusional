from flask import Flask, render_template, request
from diffusional.stable_diffusion import (
    init_pipeline,
    generate_images,
    save_generated_images,
    ImageGenerationParameters,
)
from diffusional.config import APPCONFIG

app = Flask(__name__)
pipe = init_pipeline(APPCONFIG["model"])


@app.route("/", methods=["POST", "GET"])
def home():
    model_name = APPCONFIG["model"]["name"]
    num = APPCONFIG["defaults"]["num"]
    guide = APPCONFIG["defaults"]["guide"]
    interfere = APPCONFIG["defaults"]["interfere"]
    if request.method == "POST":
        if request.form.get("generate"):
            image_generation_params = ImageGenerationParameters(request.form)
            image_files = generate_images(image_generation_params, pipe)
            return render_template(
                "index.html",
                prompt=image_generation_params.prompt,
                images=image_files,
                model_name=model_name,
                num=image_generation_params.num_of_images,
                guide=image_generation_params.guidance_scale,
                interfere=image_generation_params.interference_steps,
            )
        if request.form.get("save"):
            prefix = request.form["image-prefix"]
            assert prefix, "No image prefix provided!"
            prompt, images = save_generated_images(prefix)
            return render_template(
                "index.html",
                prompt=prompt,
                images=images,
                model_name=model_name,
                num=num,
                guide=guide,
                interfere=interfere,
            )
    return render_template(
        "index.html", model_name=model_name, num=num, guide=guide, interfere=interfere
    )
