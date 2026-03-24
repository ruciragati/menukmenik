import torch.nn as nn
from torchvision import models

def get_dog_classifier(num_classes=2):
    """
    use resnet18 model
    """
    model = models.resnet18(weights='DEFAULT')
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model