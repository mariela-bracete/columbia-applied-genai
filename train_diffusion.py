import torch
import torch.nn as nn

from helper_lib.data_loader import (
    get_cifar10_generative_data_loader,
)
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion


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

    model = get_model("diffusion")

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.network.parameters(),
        lr=1e-3,
    )

    train_diffusion(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=20,
    )


if __name__ == "__main__":
    main()