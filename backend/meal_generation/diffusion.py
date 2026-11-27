import torch
from diffusers import StableDiffusionPipeline


def load_image_model():
    # Use a CUDA GPU when available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    image_model = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=dtype,
    )
    return image_model.to(device)


def generate_meal_image(image_prompt, image_model):
    # put the meal description into  image model.
    result = image_model(image_prompt)
    return result.images[0]
