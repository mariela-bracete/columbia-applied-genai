import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )

        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, padding=1
        )

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(model_name):
    if model_name.upper() == "FCNN":
        return MLP()

    elif model_name.upper() == "CNN":
        return SimpleCNN()
    
    elif model_name.upper() == "GENERATOR":
        return Generator()

    elif model_name.upper() == "CRITIC":
        return Critic()
    
    elif model_name.upper() == "MNIST_GENERATOR":
        return MNISTGenerator()

    elif model_name.upper() == "MNIST_CRITIC":
        return MNISTCritic()

    else:
        raise ValueError(
            f"Unknown model name: {model_name}"
        )
    
class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.flatten(x)

        return x


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()

        self.z_dim = z_dim

        self.deconv1 = nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512, momentum=0.9)
        self.act1 = nn.ReLU(True)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.act2 = nn.ReLU(True)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
        self.act3 = nn.ReLU(True)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64, momentum=0.9)
        self.act4 = nn.ReLU(True)

        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), self.z_dim, 1, 1)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.act4(x)

        x = self.deconv5(x)
        x = self.tanh(x)

        return x
    
class MNISTCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.batchnorm2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class MNISTGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()

        self.z_dim = z_dim

        self.fc = nn.Linear(z_dim, 128 * 7 * 7)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(True)

        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), self.z_dim)
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)

        x = self.deconv1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)

        x = self.deconv2(x)
        x = self.tanh(x)

        return x
