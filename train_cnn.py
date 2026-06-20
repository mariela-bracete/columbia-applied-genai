import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from app.cnn_model import EnhancedCNN
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model


def main():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Using device: {device}")

    train_loader = get_data_loader(
        data_dir="./data",
        batch_size=32,
        train=True,
    )

    test_loader = get_data_loader(
        data_dir="./data",
        batch_size=32,
        train=False,
    )

    model = EnhancedCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    trained_model, datalogs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=10,
        checkpoint_dir="checkpoints/cnn",
    )

    test_loss, test_accuracy = evaluate_model(
        model=trained_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")

    os.makedirs("app/models", exist_ok=True)

    final_model_path = "app/models/cnn_cifar10.pth"
    torch.save(trained_model.state_dict(), final_model_path)

    print(f"Saved final model weights to {final_model_path}")

    best_checkpoint_dir = "checkpoints/cnn/best"
    deployed_checkpoint_path = "app/models/cnn_cifar10_checkpoint.pth"

    if os.path.isdir(best_checkpoint_dir):
        best_files = sorted(
            [
                file
                for file in os.listdir(best_checkpoint_dir)
                if file.endswith(".pth")
            ]
        )

        if best_files:
            best_checkpoint_path = os.path.join(best_checkpoint_dir, best_files[-1])
            shutil.copy(best_checkpoint_path, deployed_checkpoint_path)
            print(f"Copied best checkpoint to {deployed_checkpoint_path}")


if __name__ == "__main__":
    main()