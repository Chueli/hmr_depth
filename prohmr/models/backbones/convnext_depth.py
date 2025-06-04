"""
Simple integration using timm library - more robust and well-tested
"""
import torch
import torch.nn as nn
import timm


class ConvNeXtDepthBackbone(nn.Module):
    """
    Wrapper around timm's ConvNeXt for depth input
    """
    def __init__(self, model_name='convnext_tiny', out_features=2048, pretrained=True):
        super().__init__()
        
        # Create model with 3 input channels initially (for pretrained weights)
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'  # Use global average pooling
        )
        
        # Get the number of output features from the model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.model(dummy_input)
            in_features = features.shape[1]
        
        # Modify first conv layer for single channel input
        if hasattr(self.model, 'stem'):
            # For ConvNeXt
            original_conv = self.model.stem[0]
            self.model.stem[0] = nn.Conv2d(
                1, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Initialize new conv layer
            if pretrained:
                # Average the weights across RGB channels
                with torch.no_grad():
                    self.model.stem[0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                    if original_conv.bias is not None:
                        self.model.stem[0].bias.data = original_conv.bias.data
        
        # Add projection layer if needed
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x):
        features = self.model(x)
        return self.projection(features)