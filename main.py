from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok

import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image

import base64
from io import BytesIO


model_id = "stabilityai/stable-diffusion-xl-base-1.0"
adapter_id = "latent-consistency/lcm-lora-sdxl"

pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0).images[0]

app = Flask(__name__)
run_with_ngrok(app)

@app.route("/")
def initial():
  return render_template("index.html")

@app.route("/submit-prompt", methods=["POST"])
def generate_image():
    prompt = request.form["prompt-input"]

    image = pipe(prompt).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = "data:image/png;base64," + str(img_str)[2:-1]

    return render_template("index.html", generated_image=img_str)


if __name__ == "__main__":
    app.run()
