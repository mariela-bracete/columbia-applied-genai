import torch
import os
from tqdm import tqdm
from .checkpoints import save_checkpoint


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