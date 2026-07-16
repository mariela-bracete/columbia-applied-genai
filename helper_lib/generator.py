import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def generate_samples(model, device, num_samples=10, diffusion_steps=100):
    model.to(device)
    model.eval()

    with torch.no_grad():
        samples = model.generate(
            num_images=num_samples,
            diffusion_steps=diffusion_steps,
            image_size=32,
        ).cpu()

    grid = make_grid(samples, nrow=5, normalize=True)

    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

def generate_energy_samples(
    energy_model,
    input_images,
    steps,
    step_size,
    noise_std,
):
    """
    Generate low-energy CIFAR-10-like images using Langevin dynamics.

    Args:
        energy_model:
            Neural network that assigns one scalar energy value
            to each input image.

        input_images:
            Starting image tensor with shape
            [batch_size, 3, 32, 32].

        steps:
            Number of Langevin sampling iterations.

        step_size:
            Size of each gradient-descent update applied
            to the input images.

        noise_std:
            Standard deviation of the random noise added
            during each sampling step.

    Returns:
        Tensor of generated images with values in [-1, 1].
    """

    energy_model.eval()

    current_images = input_images.detach()

    for _ in range(steps):
        # Add random noise without tracking gradients.
        with torch.no_grad():
            noise = torch.randn_like(current_images) * noise_std

            current_images = (
                current_images + noise
            ).clamp(-1.0, 1.0)

        # Gradients are needed with respect to the images,
        # not with respect to the model parameters.
        current_images.requires_grad_(True)

        energies = energy_model(current_images)

        image_gradients, = torch.autograd.grad(
            outputs=energies,
            inputs=current_images,
            grad_outputs=torch.ones_like(energies),
        )

        # Clip gradients for sampling stability and move the
        # images in the direction of lower energy.
        with torch.no_grad():
            image_gradients = image_gradients.clamp(
                -0.03,
                0.03,
            )

            current_images = (
                current_images
                - step_size * image_gradients
            ).clamp(-1.0, 1.0)

    return current_images.detach()
