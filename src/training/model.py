import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = ("resnet18", "mobilenet_v3_small")


def create_model(model_name: str, num_classes=4, pretrained=True):
    """Create a supported transfer-learning classifier."""
    model_name = model_name.lower()
    if model_name == "resnet18":
        return _create_resnet18(num_classes=num_classes, pretrained=pretrained)
    if model_name == "mobilenet_v3_small":
        return _create_mobilenet_v3_small(num_classes=num_classes, pretrained=pretrained)

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model_name '{model_name}'. Supported: {supported}")


def _create_resnet18(num_classes, pretrained):
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _create_mobilenet_v3_small(num_classes, pretrained):
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    except AttributeError:
        model = models.mobilenet_v3_small(pretrained=pretrained)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
