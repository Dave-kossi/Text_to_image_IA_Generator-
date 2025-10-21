import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Charger le modèle (Stable Diffusion 1.5)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")  # CPU si pas de GPU (un peu plus lent)

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Image(label="Generated Image"),
    title="Text-to-Image AI Generator",
    description="Generate images from text using free AI."
)

demo.launch(share=True)
# Lancer l'interface Gradio en local