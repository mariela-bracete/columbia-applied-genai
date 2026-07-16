import os
import random

import numpy as np
import torch
from tqdm import tqdm

from .checkpoints import save_checkpoint
from .generator import generate_energy_samples


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
    checkpoint_dir="checkpoints",
):
    model.to(device)

    datalogs = []
    best_accuracy = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        model.train()

        train_loader_with_progress = tqdm(
            iterable=train_loader,
            ncols=120,
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            if batch_number % 100 == 99:
                train_loader_with_progress.set_postfix(
                    {
                        "avg accuracy": f"{running_correct / running_total:.3f}",
                        "avg loss": f"{running_loss / (batch_number + 1):.4f}",
                    }
                )

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * running_correct / running_total

        datalogs.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy,
            }
        )

        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            epoch_loss,
            epoch_accuracy,
            checkpoint_dir=checkpoint_dir,
        )

        if val_loader is not None:
            val_loss, val_accuracy = _evaluate_during_training(
                model, val_loader, criterion, device
            )

            datalogs[-1]["val_loss"] = val_loss
            datalogs[-1]["val_accuracy"] = val_accuracy

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    val_loss,
                    val_accuracy,
                    checkpoint_dir=f"{checkpoint_dir}/best",
                )
        else:
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    epoch_loss,
                    epoch_accuracy,
                    checkpoint_dir=f"{checkpoint_dir}/best",
                )

        print(
            f"Epoch {epoch + 1}: "
            f"Loss={epoch_loss:.4f}, "
            f"Accuracy={epoch_accuracy:.2f}%"
        )
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Finished Training")

    return model, datalogs


def _evaluate_during_training(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

def train_wgan(
    generator,
    critic,
    dataloader,
    opt_gen,
    opt_critic,
    device="cpu",
    z_dim=100,
    epochs=1,
    n_critic=5,
    clip_value=0.01,
    checkpoint_dir="checkpoints/gan",
    model_prefix="wgan",
):
    datalogs = []

    generator.to(device)
    critic.to(device)

    for epoch in range(epochs):
        train_loader_with_progress = tqdm(
            iterable=dataloader,
            ncols=120,
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        for batch_number, (real, _) in enumerate(train_loader_with_progress):
            real = real.to(device)
            batch_size = real.size(0)

            # Train Critic
            for _ in range(n_critic):
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake = generator(noise).detach()

                critic_real = critic(real).mean()
                critic_fake = critic(fake).mean()
                loss_critic = -(critic_real - critic_fake)

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # Train Generator
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = generator(noise)
            loss_gen = -critic(fake).mean()

            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_number % 100 == 0:
                train_loader_with_progress.set_postfix(
                    {
                        "Batch": f"{batch_number}/{len(dataloader)}",
                        "D loss": f"{loss_critic.item():.4f}",
                        "G loss": f"{loss_gen.item():.4f}",
                    }
                )

                datalogs.append(
                    {
                        "epoch": epoch + batch_number / len(dataloader),
                        "batch": batch_number / len(dataloader),
                        "critic_loss": loss_critic.item(),
                        "generator_loss": loss_gen.item(),
                    }
                )

        os.makedirs(checkpoint_dir, exist_ok=True)

    generator_path = os.path.join(checkpoint_dir, f"{model_prefix}_generator.pt")
    critic_path = os.path.join(checkpoint_dir, f"{model_prefix}_critic.pt")

    torch.save(generator.state_dict(), generator_path)
    torch.save(critic.state_dict(), critic_path)

    print(f"Generator saved to: {generator_path}")
    print(f"Critic saved to: {critic_path}")
    
    return generator, critic, datalogs

def train_diffusion(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    model.to(device)

    checkpoint_dir = "checkpoints/diffusion"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, _ in data_loader:
            images = images.to(device)
            loss = model.train_step(images, optimizer, criterion)
            total_loss += loss

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"diffusion_model_epoch_{epoch+1:03d}.pth",
        )

        torch.save(
            model.state_dict(),
            checkpoint_path,
        )

        print(
            f"Diffusion checkpoint saved: {checkpoint_path}"
        )

    final_path = os.path.join(
        checkpoint_dir,
        "diffusion_model_final.pth",
    )

    torch.save(
        model.state_dict(),
        final_path,
    )

    print(
        f"Final diffusion model saved to: {final_path}"
    )

    return model

# ---------------------------
# Assignment 4 - Energy Model
# ---------------------------

class EnergyMetric:
    """
    Tracks the running average of a scalar training metric.
    """

    def __init__(self):
        self.reset()

    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().item()

        self.total += value
        self.count += 1

    def result(self):
        if self.count == 0:
            return 0.0

        return self.total / self.count

    def reset(self):
        self.total = 0.0
        self.count = 0


class EnergyBuffer:
    """
    Replay buffer containing previously generated CIFAR-10 samples.

    Most training samples are drawn from the buffer so Langevin
    sampling does not always have to start from completely random noise.
    """

    def __init__(
        self,
        model,
        device,
        batch_size=128,
        buffer_size=8192,
        new_sample_probability=0.05,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.new_sample_probability = new_sample_probability

        # CIFAR-10 images have shape [3, 32, 32].
        # Values begin uniformly distributed in [-1, 1].
        self.examples = [
            torch.rand(
                (1, 3, 32, 32),
                device=self.device,
            )
            * 2
            - 1
            for _ in range(batch_size)
        ]

    def sample_new_examples(
        self,
        steps,
        step_size,
        noise_std,
        batch_size=None,
    ):
        if batch_size is None:
            batch_size = self.batch_size

        # Approximately 5% of the batch starts from fresh random noise.
        number_new = np.random.binomial(
            batch_size,
            self.new_sample_probability,
        )

        number_old = batch_size - number_new

        image_batches = []

        if number_new > 0:
            new_random_images = (
                torch.rand(
                    (number_new, 3, 32, 32),
                    device=self.device,
                )
                * 2
                - 1
            )

            image_batches.append(new_random_images)

        if number_old > 0:
            old_images = torch.cat(
                random.choices(
                    self.examples,
                    k=number_old,
                ),
                dim=0,
            )

            image_batches.append(old_images)

        input_images = torch.cat(image_batches, dim=0)

        # Use Langevin dynamics to move the images toward
        # lower-energy regions learned by the model.
        generated_images = generate_energy_samples(
            energy_model=self.model,
            input_images=input_images,
            steps=steps,
            step_size=step_size,
            noise_std=noise_std,
        )

        # Add new samples to the front of the replay buffer.
        self.examples = (
            list(torch.split(generated_images, 1, dim=0))
            + self.examples
        )

        # Prevent the buffer from growing indefinitely.
        self.examples = self.examples[:self.buffer_size]

        return generated_images


class EBM:
    """
    Training wrapper for the CIFAR-10 energy-based model.

    The model learns to assign:
        lower energy to real CIFAR-10 images
        higher energy to generated images
    """

    def __init__(
        self,
        model,
        alpha,
        steps,
        step_size,
        noise_std,
        device,
        batch_size=128,
    ):
        self.model = model
        self.device = device

        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise_std = noise_std

        self.buffer = EnergyBuffer(
            model=self.model,
            device=self.device,
            batch_size=batch_size,
        )

        self.loss_metric = EnergyMetric()
        self.regularization_metric = EnergyMetric()
        self.contrastive_divergence_metric = EnergyMetric()
        self.real_energy_metric = EnergyMetric()
        self.fake_energy_metric = EnergyMetric()

    def metrics(self):
        return {
            "loss": self.loss_metric.result(),
            "regularization": self.regularization_metric.result(),
            "contrastive_divergence":
                self.contrastive_divergence_metric.result(),
            "real_energy": self.real_energy_metric.result(),
            "fake_energy": self.fake_energy_metric.result(),
        }

    def reset_metrics(self):
        metrics = [
            self.loss_metric,
            self.regularization_metric,
            self.contrastive_divergence_metric,
            self.real_energy_metric,
            self.fake_energy_metric,
        ]

        for metric in metrics:
            metric.reset()

    def train_step(self, real_images, optimizer):
        self.model.train()

        real_images = real_images.to(self.device)

        # Slightly perturb the real images for more stable training.
        real_images = real_images + (
            torch.randn_like(real_images) * self.noise_std
        )

        real_images = real_images.clamp(-1.0, 1.0)

        # Generate negative examples through Langevin sampling.
        fake_images = self.buffer.sample_new_examples(
            steps=self.steps,
            step_size=self.step_size,
            noise_std=self.noise_std,
            batch_size=real_images.size(0),
        )

        # Fake samples should be treated as fixed inputs while
        # training the energy model's parameters.
        fake_images = fake_images.detach()

        input_images = torch.cat(
            [real_images, fake_images],
            dim=0,
        )

        output_energies = self.model(input_images)

        real_energies, fake_energies = torch.split(
            output_energies,
            [
                real_images.size(0),
                fake_images.size(0),
            ],
            dim=0,
        )

        # Minimize real-image energy and maximize fake-image energy.
        contrastive_divergence_loss = (
            real_energies.mean()
            - fake_energies.mean()
        )

        # Keep energy values from becoming excessively large.
        regularization_loss = self.alpha * (
            real_energies.pow(2).mean()
            + fake_energies.pow(2).mean()
        )

        loss = (
            contrastive_divergence_loss
            + regularization_loss
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=0.1,
        )

        optimizer.step()

        self.loss_metric.update(loss)
        self.regularization_metric.update(
            regularization_loss
        )
        self.contrastive_divergence_metric.update(
            contrastive_divergence_loss
        )
        self.real_energy_metric.update(
            real_energies.mean()
        )
        self.fake_energy_metric.update(
            fake_energies.mean()
        )

        return self.metrics()

    def test_step(self, real_images):
        self.model.eval()

        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)

        random_images = (
            torch.rand(
                (batch_size, 3, 32, 32),
                device=self.device,
            )
            * 2
            - 1
        )

        input_images = torch.cat(
            [real_images, random_images],
            dim=0,
        )

        with torch.no_grad():
            output_energies = self.model(input_images)

            real_energies, fake_energies = torch.split(
                output_energies,
                [batch_size, batch_size],
                dim=0,
            )

            contrastive_divergence = (
                real_energies.mean()
                - fake_energies.mean()
            )

        return {
            "contrastive_divergence":
                contrastive_divergence.item(),
            "real_energy":
                real_energies.mean().item(),
            "fake_energy":
                fake_energies.mean().item(),
        }


def train_energy(
    model,
    train_loader,
    optimizer,
    device="cpu",
    epochs=10,
    alpha=0.1,
    langevin_steps=60,
    step_size=10.0,
    noise_std=0.005,
    checkpoint_dir="checkpoints/energy",
):
    """
    Train an energy-based model on CIFAR-10 images.
    """

    model.to(device)

    ebm = EBM(
        model=model,
        alpha=alpha,
        steps=langevin_steps,
        step_size=step_size,
        noise_std=noise_std,
        device=device,
        batch_size=train_loader.batch_size,
    )

    training_logs = []

    for epoch in range(epochs):
        ebm.reset_metrics()

        progress_bar = tqdm(
            iterable=train_loader,
            ncols=120,
            desc=f"Energy Epoch {epoch + 1}/{epochs}",
        )

        for real_images, _ in progress_bar:
            metrics = ebm.train_step(
                real_images=real_images,
                optimizer=optimizer,
            )

            progress_bar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "real": (
                        f"{metrics['real_energy']:.4f}"
                    ),
                    "fake": (
                        f"{metrics['fake_energy']:.4f}"
                    ),
                }
            )

        epoch_metrics = ebm.metrics()

        training_logs.append(
            {
                "epoch": epoch + 1,
                **epoch_metrics,
            }
        )

        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"energy_model_epoch_{epoch + 1:03d}.pth",
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "metrics": epoch_metrics,
            },
            checkpoint_path,
        )

        print(
            f"Epoch {epoch + 1}: "
            f"Loss={epoch_metrics['loss']:.4f}, "
            f"Real energy="
            f"{epoch_metrics['real_energy']:.4f}, "
            f"Fake energy="
            f"{epoch_metrics['fake_energy']:.4f}"
        )

        print(
            f"Energy checkpoint saved: "
            f"{checkpoint_path}"
        )

    final_checkpoint_path = os.path.join(
        checkpoint_dir,
        "energy_model_final.pth",
    )

    torch.save(
        model.state_dict(),
        final_checkpoint_path,
    )

    print(
        f"Final energy model saved to: "
        f"{final_checkpoint_path}"
    )

    return model, ebm, training_logs