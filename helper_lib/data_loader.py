import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


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

from PIL import Image
import os


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0
    
def get_gan_data_loader(
    image_dir="./data/celeba/img_align_celeba",
    batch_size=128,
    shuffle=True,
):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = CustomImageDataset(
        root_dir=image_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

    return loader

# ---------------------------
# Assignment 3 - MNIST GAN
# ---------------------------

def get_mnist_gan_data_loader(
    data_dir="./data",
    batch_size=128,
    train=True,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
    )

    return loader
