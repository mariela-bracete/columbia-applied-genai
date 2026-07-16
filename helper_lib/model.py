import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


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

def offset_cosine_diffusion_schedule(
    diffusion_times,
    min_signal_rate=0.02,
    max_signal_rate=0.95,
):
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    start_angle = torch.acos(
        torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device)
    )
    end_angle = torch.acos(
        torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device)
    )

    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)

    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)

    return noise_rates, signal_rates


class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies)
        )
        self.register_buffer(
            "angular_speeds",
            2.0 * math.pi * frequencies.view(1, 1, 1, -1),
        )

    def forward(self, x):
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.proj(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()

        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width

        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)

        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()

        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return x


class UNet(nn.Module):
    def __init__(self, image_size, num_channels, embedding_dim=32):
        super().__init__()

        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim

        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

        self.down1 = DownBlock(32, in_channels=64, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2)

        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

        self.up1 = UpBlock(96, in_channels=128, block_depth=2)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []

        x = self.initial(noisy_images)

        noise_emb = self.embedding(noise_variances)
        noise_emb = F.interpolate(
            noise_emb.permute(0, 3, 1, 2),
            size=(self.image_size, self.image_size),
            mode="nearest",
        )

        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)


class DiffusionModel(nn.Module):
    def __init__(self, model, schedule_fn):
        super().__init__()

        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_network.eval()

        self.ema_decay = 0.8
        self.schedule_fn = schedule_fn

        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def to(self, device):
        super().to(device)
        self.ema_network.to(device)
        return self

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, x):
        return torch.clamp(
            x * self.normalizer_std + self.normalizer_mean,
            0.0,
            1.0,
        )

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()

        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise

        for step in range(diffusion_steps):
            t = torch.ones(
                (initial_noise.shape[0], 1, 1, 1),
                device=initial_noise.device,
            ) * (1 - step * step_size)

            noise_rates, signal_rates = self.schedule_fn(t)

            pred_noises, pred_images = self.denoise(
                current_images,
                noise_rates,
                signal_rates,
                training=False,
            )

            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)

            current_images = (
                next_signal_rates * pred_images
                + next_noise_rates * pred_noises
            )

        return pred_images

    def generate(self, num_images, diffusion_steps, image_size=64, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn(
                (
                    num_images,
                    self.network.num_channels,
                    image_size,
                    image_size,
                ),
                device=next(self.parameters()).device,
            )

        with torch.no_grad():
            return self.denormalize(
                self.reverse_diffusion(initial_noise, diffusion_steps)
            )

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand(
            (images.size(0), 1, 1, 1),
            device=images.device,
        )

        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self.denoise(
            noisy_images,
            noise_rates,
            signal_rates,
            training=True,
        )

        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the exponential moving average network.
        # The EMA network is used when generating images.
        with torch.no_grad():
            for ema_parameter, parameter in zip(
                self.ema_network.parameters(),
                self.network.parameters(),
            ):
                ema_parameter.mul_(self.ema_decay)
                ema_parameter.add_(
                    parameter,
                    alpha=1.0 - self.ema_decay,
                )

        return loss.item()

    def test_step(self, images, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand(
            (images.size(0), 1, 1, 1),
            device=images.device,
        )

        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with torch.no_grad():
            pred_noises, _ = self.denoise(
                noisy_images,
                noise_rates,
                signal_rates,
                training=False,
            )
            loss = loss_fn(pred_noises, noises)

        return loss.item()
    
class EnergyModel(nn.Module):
    """
    Energy-based model for CIFAR-10 images.

    Input shape:
        [batch_size, 3, 32, 32]

    Output shape:
        [batch_size, 1]

    The single output value is the learned energy assigned
    to each input image.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2,
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            64 * 2 * 2,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            1,
        )

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = F.silu(self.conv4(x))

        x = self.flatten(x)

        x = F.silu(self.fc1(x))
        energy = self.fc2(x)

        return energy    

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
    
    elif model_name.upper() == "ENERGY":
        return EnergyModel()
    
    elif model_name.lower() == "diffusion":
        unet = UNet(
            image_size=32,
            num_channels=3,
            embedding_dim=32,
        )

        model = DiffusionModel(
            unet,
            offset_cosine_diffusion_schedule,
        )

        return model

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
