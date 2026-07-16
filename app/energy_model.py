import io

import torch
from fastapi import Response
from PIL import Image

from helper_lib.generator import generate_energy_samples
from helper_lib.model import get_model


CHECKPOINT_PATH = "checkpoints/energy/energy_model_final.pth"


def generate_energy_image():
    """
    Generate one CIFAR-10-style image using the trained
    Energy-Based Model and return it as a PNG.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Instantiate the model
    model = get_model("ENERGY")
    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    # Start Langevin sampling from random noise
    initial_image = (
        torch.rand(
            (1, 3, 32, 32),
            device=device,
        )
        * 2
        - 1
    )

    generated_image = generate_energy_samples(
        energy_model=model,
        input_images=initial_image,
        steps=60,
        step_size=10.0,
        noise_std=0.005,
    )

    image = generated_image.squeeze(0)

    # Convert from [-1, 1] back to [0, 1]
    image = (image + 1.0) / 2.0
    image = image.clamp(0.0, 1.0)

    image = image.permute(1, 2, 0)
    image = (image * 255).byte().cpu().numpy()

    pil_image = Image.fromarray(image)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
    )