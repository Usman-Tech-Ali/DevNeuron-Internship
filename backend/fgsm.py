# fgsm.py
# Implementation of Fast Gradient Sign Method (FGSM) attack in PyTorch.
# Adapted from the official PyTorch tutorial: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
# This encapsulates the FGSM attack in an Attack class for modularity.
# Assumes input image is a grayscale MNIST-like tensor of shape (1, 1, 28, 28) with values in [0, 1].
# For untargeted attack (default), uses the model's prediction on the clean image as the target label
# to fool (maximizes loss for that class). For targeted, minimizes loss for the provided target_label.
# Reference: Goodfellow et al. (2014) - Explaining and Harnessing Adversarial Examples (arXiv:1412.6572).

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attack:
    """
    FGSM Attack class.
    
    Args:
        epsilon (float): Perturbation magnitude. Default: 0.1.
    """
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def generate(self, model: nn.Module, image: torch.Tensor, true_label: torch.Tensor, target_label: int = None) -> torch.Tensor:
        """
        Generates an adversarial example using FGSM.
        
        Args:
            model (nn.Module): Pretrained model (e.g., for MNIST classification).
            image (torch.Tensor): Input image tensor of shape (1, 1, 28, 28), values in [0, 1].
            true_label (torch.Tensor): True label tensor for untargeted attack.
            target_label (int, optional): Target class for targeted attack. If None, performs untargeted attack.
        
        Returns:
            torch.Tensor: Adversarial image tensor of same shape as input, clipped to [0, 1].
        """
        # Ensure image requires gradients
        image = image.clone().detach().requires_grad_(True)
        
        # Forward pass to get logits
        output = model(image)
        
        # Determine target label
        if target_label is None:
            # Untargeted: use true label to fool
            target = true_label
        else:
            target = torch.tensor([target_label], device=image.device)
        
        # Compute loss (cross-entropy)
        loss = F.cross_entropy(output, target)
        
        # Compute gradients
        model.zero_grad()
        loss.backward()
        # Get sign of gradients w.r.t. image
        gradient = image.grad.data
        
        # Create perturbation: for untargeted (maximize loss), +sign(gradient);
        # for targeted (minimize loss), -sign(gradient)
        if target_label is None:
            perturbation = self.epsilon * gradient.sign()
        else:
            perturbation = -self.epsilon * gradient.sign()
        
        # Generate adversarial image
        adversarial_image = image + perturbation
        
        # Clip to [0, 1]
        adversarial_image = torch.clamp(adversarial_image, 0, 1)
        
        # Detach from computation graph
        return adversarial_image.detach()

# Simple test function
def test_fgsm():
    """
    Simple test: Define a dummy MNIST-like model (LeNet variant), create a random image,
    run untargeted and targeted FGSM, and print predictions.
    """
    # Dummy LeNet model for MNIST (10 classes)
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

    # Instantiate model and attack
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMNISTModel().to(device)
    attack = Attack(epsilon=0.1)

    # Dummy random image (simulate a '5'-like but random for test)
    image = torch.rand(1, 1, 28, 28, device=device)
    true_label = torch.tensor([5], device=device)

    # Untargeted attack
    adv_untargeted = attack.generate(model, image, true_label)
    output_untargeted = model(image)
    output_adv_untargeted = model(adv_untargeted)
    _, clean_label = torch.max(output_untargeted, 1)
    _, adv_label_untargeted = torch.max(output_adv_untargeted, 1)
    print(f"Untargeted - Clean label: {clean_label.item()}, Adv label: {adv_label_untargeted.item()}")

    # Targeted attack (target label 3)
    adv_targeted = attack.generate(model, image, true_label, target_label=3)
    output_adv_targeted = model(adv_targeted)
    _, adv_label_targeted = torch.max(output_adv_targeted, 1)
    print(f"Targeted (target=3) - Adv label: {adv_label_targeted.item()}")

    print("Test completed. Adversarial images shapes:", adv_untargeted.shape, adv_targeted.shape)

if __name__ == "__main__":
    test_fgsm()