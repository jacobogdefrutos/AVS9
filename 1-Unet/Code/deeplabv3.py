from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn as nn


def createDeepLabv3(inputchannels=3, outputchannels=2):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(weights=None,progress=True,aux_loss=True)#weights=DeepLabV3_ResNet101_Weights.DEFAULT,
                                                    
    #model.classifier = DeepLabHead(2048, outputchannels)
    model.classifier[4] = nn.Conv2d(256, outputchannels, 1)
    model.aux_classifier[4] = nn.Conv2d(256, outputchannels, 1)
    #for param in model.backbone.parameters():
    #    param.requires_grad = False
    # Set the model in training mode
    #model.train()
    return model

def createDeepLabv3_1(inputchannels=3, outputchannels=2):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT,
                                                    progress=True)
        # Freeze all layers except model.classifier[4] and model.aux_classifier[4]
    for name, param in model.named_parameters():
        if 'classifier.4' in name or 'aux_classifier.4' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    model.classifier[4] = nn.Conv2d(256, outputchannels, 1)
    model.aux_classifier[4] = nn.Conv2d(256, outputchannels, 1)
    
    return model