import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_data_loader(data_dir="./data", batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return loader