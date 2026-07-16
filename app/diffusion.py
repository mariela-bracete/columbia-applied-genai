import io

import torch
from PIL import Image

from helper_lib.model import get_model
from helper_lib.generator import generate_samples


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


model = get_model("diffusion")

checkpoint = torch.load(
    "checkpoints/diffusion/diffusion_model_final.pth",
    map_location=DEVICE,
)

model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()


def generate_diffusion_image():

    with torch.no_grad():

        samples = model.generate(
            num_images=1,
            diffusion_steps=100,
            image_size=32,
        ).cpu()

    image = samples[0]

    image = (
        image.clamp(-1, 1)
        + 1
    ) / 2

    image = (
        image.permute(1, 2, 0)
        .numpy()
        * 255
    ).astype("uint8")

    pil_image = Image.fromarray(image)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")

    return buffer.getvalue()