import base64
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from helper_lib.model import MNISTGenerator
from fastapi.responses import Response


Z_DIM = 100
CHECKPOINT_PATH = Path("app/models/mnist_wgan_generator.pt")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

mnist_generator = MNISTGenerator(z_dim=Z_DIM).to(device)


def load_mnist_generator():
    if CHECKPOINT_PATH.exists():
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        mnist_generator.load_state_dict(state_dict)
        mnist_generator.eval()
        return True

    mnist_generator.eval()
    return False


checkpoint_loaded = load_mnist_generator()


def generate_mnist_digit():
    noise = torch.randn(1, Z_DIM, 1, 1).to(device)

    with torch.no_grad():
        generated = mnist_generator(noise).detach().cpu().squeeze(0)

    # Undo tanh normalization: [-1, 1] -> [0, 1]
    generated = (generated * 0.5) + 0.5
    generated = generated.clamp(0, 1)

    # Convert tensor shape from [1, 28, 28] to a grayscale PIL image
    array = (generated.squeeze(0).numpy() * 255).astype("uint8")
    image = Image.fromarray(array, mode="L")

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "X-Checkpoint-Loaded": str(checkpoint_loaded)
        },
    )