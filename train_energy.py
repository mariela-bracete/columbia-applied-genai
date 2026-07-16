import torch

from helper_lib.data_loader import (
    get_cifar10_generative_data_loader,
)
from helper_lib.model import get_model
from helper_lib.trainer import train_energy


def main():
    device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

    print(f"Using device: {device}")

    train_loader = get_cifar10_generative_data_loader(
        batch_size=128,
        train=True,
    )

    model = get_model("ENERGY")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    train_energy(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=20,
        alpha=0.1,
        langevin_steps=60,
        step_size=10.0,
        noise_std=0.005,
        checkpoint_dir="checkpoints/energy",
    )


if __name__ == "__main__":
    main()