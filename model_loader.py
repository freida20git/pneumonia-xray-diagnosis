import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from torchvision import models
import torch.nn.functional as F

# Define the model architecture (Must be same as the trained model)
class DenseEfficientNet(nn.Module):
    def __init__(self, num_classes=1):
        super(DenseEfficientNet, self).__init__()

        # Load pre-trained DenseNet121
        self.densenet = models.densenet161(pretrained=True)
        dense_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()  # Remove DenseNet head

        # Load pre-trained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        eff_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()  # Remove EfficientNet head

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(dense_features + eff_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        dense_features = self.densenet(x)
        efficient_features = self.efficientnet(x)
        combined_features = torch.cat((dense_features, efficient_features), dim=1)
        output = self.classifier(combined_features)
        return output


# Load model structure
model = DenseEfficientNet()

# Load trained weights
model.load_state_dict(torch.load(r"C:\Users\fried\Desktop\School 2025\Deep Learning\project 2 Xray\trained_model_20epochs_1.pth", map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode
print("loaded model succesfully")

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        self.hooks = []
        def hook_fn(module, input, output):
            self.activations = output
            output.register_hook(self.save_gradient)

        # Ensure the target layer exists and register the hook correctly
        flag =False
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.hooks.append(module.register_forward_hook(hook_fn))
                print("   found layer!! ")
                flag = True
        if not flag:
            raise ValueError(f"Target layer '{self.target_layer_name}' not found in the model.")

        output = self.model(x)
        return output

    def generate_cam(self, class_idx):
        gradients = self.gradients
        activations = self.activations

        # Check if gradients are available before calculating mean
        if gradients is None:
            raise ValueError("Gradients are not available. Make sure to call 'forward' before 'generate_cam'.")

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(len(pooled_gradients)):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap - torch.min(heatmap)
        heatmap = heatmap / torch.max(heatmap)
        heatmap = heatmap.cpu().detach().numpy()

        return heatmap

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
print("gradCAM defined")
