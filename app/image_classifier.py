from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.cnn_model import EnhancedCNN


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


MODEL_PATH = Path(__file__).parent / "models" / "cnn_cifar10.pth"

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

model = EnhancedCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def classify_image(image: Image.Image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASSES[predicted_idx.item()]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence.item(),
    }