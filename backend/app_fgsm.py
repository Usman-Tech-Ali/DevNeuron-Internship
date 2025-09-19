# app_fgsm.py
# FastAPI service for FGSM adversarial attack on MNIST model.
# Endpoint: POST /attack - accepts image upload and epsilon, returns predictions and base64 adv image.
# Integrates Attack from fgsm.py and a simple pretrained MNIST model (trained on startup).
# Preprocessing: Resize to 28x28 grayscale, tensor in [0,1].
# Reference: FastAPI file upload docs - https://fastapi.tiangolo.com/tutorial/request-files/
#           PyTorch MNIST example - https://pytorch.org/tutorials/basics/data_tutorial.html
# Assumes fgsm.py in same directory. Run with: uvicorn app_fgsm:app --reload

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import io
import base64
from fgsm import Attack  # Import Attack class

# Define simple MNIST model
class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Global model and attack (trained once at startup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMNISTModel().to(device)
attack = Attack(epsilon=0.1)  # Default epsilon, overridden per request

# Quick training function (called once)
def train_model():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model.train()
    print("Training model (1 epoch)...")
    for epoch in range(1):  # Quick 1 epoch for demo
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    model.eval()
    print("Model trained and ready.")

# Train on startup
train_model()

app = FastAPI(title="FGSM Adversarial Attack API")

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Convert uploaded image to MNIST-format tensor (1,1,28,28) in [0,1]."""
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Grayscale
    image = image.resize((28, 28))
    transform = transforms.ToTensor()
    tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim
    return tensor

def predict(model: nn.Module, image: torch.Tensor) -> tuple[int, float]:
    """Get top prediction label and confidence."""
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        confidence, label = torch.max(probs, 1)
        return label.item(), confidence.item()

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert tensor to base64 PNG string."""
    # Denormalize to PIL Image
    img = transforms.ToPILImage()(tensor.squeeze().cpu())
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.post("/attack")
async def attack_endpoint(image: UploadFile = File(..., description="Image file (PNG/JPG)"),
                          epsilon: float = Form(0.1, description="Perturbation epsilon")):
    """FGSM attack endpoint."""
    if not image.content_type.startswith('image/'):
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})
    
    # Read and preprocess
    image_bytes = await image.read()
    clean_image = preprocess_image(image_bytes)
    attack.epsilon = epsilon  # Update epsilon for this request
    
    # Clean prediction
    clean_label, clean_conf = predict(model, clean_image)
    clean_pred_str = f"{clean_label} ({clean_conf:.2f})"
    
    # Generate adversarial (untargeted attack)
    true_label_tensor = torch.tensor([clean_label], device=device)
    adv_image = attack.generate(model, clean_image, true_label_tensor, target_label=None)  # Untargeted
    
    # Adv prediction
    adv_label, adv_conf = predict(model, adv_image)
    adv_pred_str = f"{adv_label} ({adv_conf:.2f})"
    
    # Base64 adv image
    base64_adv = tensor_to_base64(adv_image)
    
    # Success: labels differ
    success = clean_label != adv_label
    
    return {
        "clean_prediction": clean_pred_str,
        "adversarial_prediction": adv_pred_str,
        "base64_adversarial_image": base64_adv,
        "attack_success": success
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)