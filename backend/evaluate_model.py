# evaluate_model.py
# Script to evaluate the robustness of a pretrained MNIST model using FGSM attacks.
# Trains a simple CNN quickly (1 epoch for demo; increase for better accuracy).
# Evaluates on 100 samples from MNIST test set.
# Computes clean and adversarial accuracy, records drop.
# Outputs to console and accuracy_drop.txt.
# Reference: PyTorch MNIST loader - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# Adapted FGSM from official tutorial: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
# Note: Requires fgsm.py in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fgsm import Attack  # Import the Attack class from previous file

# Define simple MNIST model (LeNet-like)
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

# Quick training (1 epoch; for demo - expect ~90%+ clean acc)
def train_model(model, train_loader, epochs=1):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} completed.')

# Evaluation on num_samples
def evaluate(model, test_loader, attack, num_samples=100):
    model.eval()
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    for data, target in test_loader:
        if total >= num_samples:
            break
        data, target = data.to(device), target.to(device)
        
        # Clean prediction (no gradients needed)
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)
            correct_clean += pred.eq(target).sum().item()
        
        # Adversarial generation (gradients needed)
        data_requires_grad = data.clone().detach().requires_grad_(True)
        adv_data = attack.generate(model, data_requires_grad, target, target_label=None)  # untargeted
        
        # Adversarial prediction (no gradients needed)
        with torch.no_grad():
            output_adv = model(adv_data)
            pred_adv = output_adv.argmax(dim=1)
            correct_adv += pred_adv.eq(target).sum().item()
        
        total += data.size(0)
    clean_acc = 100. * correct_clean / total
    adv_acc = 100. * correct_adv / total
    drop = clean_acc - adv_acc
    return clean_acc, adv_acc, drop

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading (no normalization for simple [0,1] FGSM clamping)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model and attack
    model = SimpleMNISTModel().to(device)
    print("Training model (1 epoch)...")
    train_model(model, train_loader, epochs=1)
    attack = Attack(epsilon=0.3)  # Increased epsilon for stronger attacks

    # Evaluate
    print("Evaluating on 100 test samples...")
    clean_acc, adv_acc, drop = evaluate(model, test_loader, attack, num_samples=100)

    # Print results
    print(f"\nClean accuracy: {clean_acc:.2f}%")
    print(f"Adversarial accuracy: {adv_acc:.2f}%")
    print(f"Accuracy drop: {drop:.2f}%")
    print("Screenshot this console output for submission.")

    # Write to file
    with open('accuracy_drop.txt', 'w') as f:
        f.write(f"Model: Simple CNN on MNIST\n")
        f.write(f"Samples: 100\n")
        f.write(f"Epsilon: 0.3\n")
        f.write(f"Clean accuracy: {clean_acc:.2f}%\n")
        f.write(f"Adversarial accuracy: {adv_acc:.2f}%\n")
        f.write(f"Accuracy drop: {drop:.2f}%\n")
    print("\nResults saved to accuracy_drop.txt")