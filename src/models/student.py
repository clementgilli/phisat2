import torch
import torch.nn as nn
from typing import Any, Dict, List

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks."""
    
    def __init__(self, drop_prob: float = 0.0):
        """
        Initialize DropPath.
        
        Args:
            drop_prob (float): Drop probability. Defaults to 0.0.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with depthwise convolution, batch norm, and inverted bottleneck."""
    
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        """
        Initialize ConvNeXtBlock.
        
        Args:
            dim (int): Number of input/output channels.
            drop_path (float): Drop path rate. Defaults to 0.0.
            layer_scale_init_value (float): Layer scale initialization value. Defaults to 1e-6.
        """
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)  # pointwise/1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        input_tensor = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        
        x = input_tensor + self.drop_path(x)
        return x


class ConvBlock(nn.Module):
    """ConvNeXt-style block with BatchNorm for improved performance."""
    
    def __init__(self, in_channels: int, out_channels: int, drop_path: float = 0.0):
        """
        Initialize ConvBlock with ConvNeXt architecture using BatchNorm.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            drop_path (float): Drop path rate. Defaults to 0.0.
        """
        super().__init__()
        
        # If input and output channels differ, use a 1x1 conv to match dimensions
        if in_channels != out_channels:
            self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.channel_proj = nn.Identity()
        
        # ConvNeXt block
        self.convnext_block = ConvNeXtBlock(out_channels, drop_path=drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.channel_proj(x)
        x = self.convnext_block(x)
        return x

class PhisatNetEncoder(nn.Module):
    """
    Backbone partagé. Prend une image et ressort un dictionnaire de features (skips + bottleneck).
    """
    def __init__(self, 
                 n_channels: int = 8, 
                 base_filters: int = 32,
                 depth: int = 3,
                 channel_multipliers: List[int] = None):
        super().__init__()
        self.depth = depth
        if channel_multipliers is None:
            channel_multipliers = [2**i for i in range(depth + 1)]
            
        self.channels = [base_filters * mult for mult in channel_multipliers]
        
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        self.encoders.append(ConvBlock(n_channels, self.channels[0]))
        
        for i in range(depth - 1):
            self.pools.append(nn.MaxPool2d(2))
            self.encoders.append(ConvBlock(self.channels[i], self.channels[i + 1]))
            
        self.pools.append(nn.MaxPool2d(2))
        self.bottleneck = ConvBlock(self.channels[depth - 1], self.channels[depth])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        current = x
        
        for i in range(self.depth):
            current = self.encoders[i](current)
            features[f'enc_{i}'] = current
            
            if i < self.depth - 1:
                current = self.pools[i](current)
                
        current = self.pools[-1](current)
        features['bottleneck'] = self.bottleneck(current)
        
        return features
    
class PhisatNetDecoder(nn.Module):
    
    def __init__(self, 
                 n_classes: int = 3, 
                 base_filters: int = 32,
                 depth: int = 3,
                 channel_multipliers: List[int] = None):
        super().__init__()
        self.depth = depth
        if channel_multipliers is None:
            channel_multipliers = [2**i for i in range(depth + 1)]
            
        self.channels = [base_filters * mult for mult in channel_multipliers]
        
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(depth):
            up_in_channels = self.channels[depth - i]
            up_out_channels = self.channels[depth - i]
            self.upsamplers.append(
                nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2)
            )
            
            dec_in_channels = self.channels[depth - i] + self.channels[depth - i - 1]
            dec_out_channels = self.channels[depth - i - 1]
            self.decoders.append(ConvBlock(dec_in_channels, dec_out_channels))
            
        self.final_conv = nn.Conv2d(self.channels[0], n_classes, kernel_size=1)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        current = features['bottleneck']
        
        for i in range(self.depth):
            current = self.upsamplers[i](current)
            
            skip_connection = features[f'enc_{self.depth - 1 - i}']
            current = torch.cat([current, skip_connection], dim=1)
            
            current = self.decoders[i](current)
            
        return self.final_conv(current)

class GlobalHeadDecoder(nn.Module):
    
    def __init__(self, in_channels: int, out_features: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(in_channels, out_features)
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = features['bottleneck']
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)