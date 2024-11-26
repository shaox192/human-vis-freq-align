
# import torch
import torch.nn as nn
from torchvision import models as torchmodels



def get_model_weights(arch):
    if arch == "resnet18":
        return torchmodels.ResNet18_Weights.DEFAULT
    elif arch == "resnet50":
        return torchmodels.ResNet50_Weights.DEFAULT
    else:
        raise NotImplementedError


def get_classifier(arch, num_classes, pretrained=True):
    if pretrained:
        if num_classes != 1000:
            pretrained_classifier = torchmodels.__dict__[arch](weights=get_model_weights(arch))
            pretrained_classifier.fc = nn.Linear(pretrained_classifier.fc.in_features, num_classes)
            return pretrained_classifier
        
        classifier = torchmodels.__dict__[arch](num_classes=num_classes,
                                                weights=get_model_weights(arch))
    else:
        
        classifier = torchmodels.__dict__[arch](num_classes=num_classes)
    
    return classifier
