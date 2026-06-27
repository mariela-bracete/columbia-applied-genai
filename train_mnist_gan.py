import os
import shutil

import torch
import torch.optim as optim

from helper_lib.data_loader import get_mnist_gan_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_wgan


def main():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print(f"Using device: {device}")

    z_dim = 100
    batch_size = 128
    lr = 5e-5
    epochs = 10
    n_critic = 5
    clip_value = 0.01

    dataloader = get_mnist_gan_data_loader(
        data_dir="./data",
        batch_size=batch_size,
        train=True,
    )

    generator = get_model("MNIST_GENERATOR").to(device)
    critic = get_model("MNIST_CRITIC").to(device)

    opt_gen = optim.RMSprop(generator.parameters(), lr=lr)
    opt_critic = optim.RMSprop(critic.parameters(), lr=lr)

    trained_generator, trained_critic, datalogs = train_wgan(
        generator=generator,
        critic=critic,
        dataloader=dataloader,
        opt_gen=opt_gen,
        opt_critic=opt_critic,
        device=device,
        z_dim=z_dim,
        epochs=epochs,
        n_critic=n_critic,
        clip_value=clip_value,
        checkpoint_dir="checkpoints/mnist_gan",
        model_prefix="mnist_wgan",
    )

    os.makedirs("app/models", exist_ok=True)

    final_generator_path = "app/models/mnist_wgan_generator.pt"
    torch.save(trained_generator.state_dict(), final_generator_path)

    print(f"Saved final MNIST generator weights to {final_generator_path}")

    checkpoint_generator_path = "checkpoints/mnist_gan/mnist_wgan_generator.pt"

    if os.path.exists(checkpoint_generator_path):
        shutil.copy(checkpoint_generator_path, final_generator_path)
        print(f"Copied generator checkpoint to {final_generator_path}")


if __name__ == "__main__":
    main()