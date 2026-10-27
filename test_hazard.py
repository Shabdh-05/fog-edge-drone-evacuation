import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

classes = ['cyclone', 'earthquake', 'flood', 'wildfire']

class HazardCNN(nn.Module):
    def __init__(self):
        super(HazardCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = HazardCNN().to(device)
model.load_state_dict(torch.load("models/hazard_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 🔥 Using real image from your folder
image_path = "datasets/hazards/test/cyclone/500.jpg"

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print("Predicted Hazard:", classes[predicted.item()])