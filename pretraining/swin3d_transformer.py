"""
Swin3D Transformer Implementation from Scratch
Extended from Swin Transformer for 3D data (videos, medical imaging, etc.)
Paper: "Video Swin Transformer" and extensions for 3D medical imaging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


def window_partition_3d(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Partition 3D feature map into non-overlapping windows.
    
    Args:
        x: (B, D, H, W, C)
        window_size: Window size (D, H, W)
        
    Returns:
        windows: (num_windows*B, window_size[0], window_size[1], window_size[2], C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], 
               H // window_size[1], window_size[1],
               W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse_3d(windows: torch.Tensor, window_size: Tuple[int, int, int], 
                       D: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse 3D window partition back to feature map.
    
    Args:
        windows: (num_windows*B, window_size[0], window_size[1], window_size[2], C)
        window_size: Window size (D, H, W)
        D: Depth of feature map
        H: Height of feature map
        W: Width of feature map
        
    Returns:
        x: (B, D, H, W, C)
    """
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class PatchEmbedding3D(nn.Module):
    """3D Image/Video to Patch Embedding using 3D convolution"""
    
    def __init__(self, img_size: Tuple[int, int, int] = (16, 224, 224), 
                 patch_size: Tuple[int, int, int] = (2, 4, 4),
                 in_channels: int = 3, embed_dim: int = 96, 
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size[0],
                                  img_size[1] // patch_size[1],
                                  img_size[2] // patch_size[2])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1] * self.patches_resolution[2]
        
        self.proj = nn.Conv3d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, D', H', W'
        x = x.flatten(2).transpose(1, 2)  # B, D'*H'*W', embed_dim
        x = self.norm(x)
        return x


class PatchMerging3D(nn.Module):
    """3D Patch Merging Layer - downsamples by 2x in each dimension and increases channels"""
    
    def __init__(self, input_resolution: Tuple[int, int, int], dim: int, 
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "Input feature size doesn't match"
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, "Dimensions must be even"
        
        x = x.view(B, D, H, W, C)
        
        # Concatenate 2x2x2 neighboring patches
        x0 = x[:, 0::2, 0::2, 0::2, :]  # B, D/2, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B, D/2, H/2, W/2, 8*C
        x = x.view(B, -1, 8 * C)  # B, D/2*H/2*W/2, 8*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class WindowAttention3D(nn.Module):
    """3D Window-based Multi-head Self Attention with relative position bias"""
    
    def __init__(self, dim: int, window_size: Tuple[int, int, int], num_heads: int, 
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias table for 3D
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * 
                       (2 * window_size[2] - 1), num_heads)
        )
        
        # Get pair-wise relative position index for 3D
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1] * self.window_size[2],
               self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply attention mask for shifted windows
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, 
                 out_features: Optional[int] = None, act_layer: nn.Module = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """3D Swin Transformer Block with window-based or shifted window-based attention"""
    
    def __init__(self, dim: int, input_resolution: Tuple[int, int, int], num_heads: int, 
                 window_size: Tuple[int, int, int] = (8, 7, 7), 
                 shift_size: Tuple[int, int, int] = (0, 0, 0),
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0., 
                 attn_drop: float = 0., drop_path: float = 0., 
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Adjust window size and shift size if resolution is too small
        if min(self.input_resolution) <= min(self.window_size):
            self.shift_size = (0, 0, 0)
            self.window_size = self.input_resolution
            
        assert all(0 <= s < w for s, w in zip(self.shift_size, self.window_size)), \
            "shift_size must be between 0 and window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)
        
        # Create attention mask for shifted window
        if any(s > 0 for s in self.shift_size):
            D, H, W = self.input_resolution
            img_mask = torch.zeros((1, D, H, W, 1))
            d_slices = (slice(0, -self.window_size[0]),
                       slice(-self.window_size[0], -self.shift_size[0]),
                       slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]),
                       slice(-self.window_size[1], -self.shift_size[1]),
                       slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]),
                       slice(-self.window_size[2], -self.shift_size[2]),
                       slice(-self.shift_size[2], None))
            cnt = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, d, h, w, :] = cnt
                        cnt += 1
                        
            mask_windows = window_partition_3d(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, 
                                            self.window_size[0] * self.window_size[1] * self.window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            
        self.register_buffer("attn_mask", attn_mask)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "Input feature size doesn't match"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)
        
        # Cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], 
                                              -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
            
        # Partition windows
        x_windows = window_partition_3d(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, 
                                   self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], 
                                        self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, D, H, W)
        
        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], 
                                             self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        x = x.view(B, D * H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class BasicLayer3D(nn.Module):
    """A basic 3D Swin Transformer layer for one stage"""
    
    def __init__(self, dim: int, input_resolution: Tuple[int, int, int], depth: int, 
                 num_heads: int, window_size: Tuple[int, int, int], mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0., 
                 drop_path: float = 0., norm_layer: nn.Module = nn.LayerNorm, 
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else tuple(w // 2 for w in window_size),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinTransformer3D(nn.Module):
    """
    3D Swin Transformer for video/volumetric data
    
    Args:
        img_size: Input size (D, H, W) - e.g., (16, 224, 224) for video
        patch_size: Patch size (D, H, W) - e.g., (2, 4, 4)
        in_channels: Number of input channels (3 for RGB video)
        num_classes: Number of classes for classification
        embed_dim: Patch embedding dimension
        depths: Depth of each Swin Transformer layer
        num_heads: Number of attention heads in different layers
        window_size: Window size (D, H, W)
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: If True, add a learnable bias to query, key, value
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        norm_layer: Normalization layer
    """
    
    def __init__(self, img_size: Tuple[int, int, int] = (16, 224, 224), 
                 patch_size: Tuple[int, int, int] = (2, 4, 4),
                 in_channels: int = 3, num_classes: int = 400, embed_dim: int = 96, 
                 depths: Tuple[int] = (2, 2, 6, 2), num_heads: Tuple[int] = (3, 6, 12, 24),
                 window_size: Tuple[int, int, int] = (8, 7, 7), mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, norm_layer: nn.Module = nn.LayerNorm, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # Split 3D input into non-overlapping patches
        self.patch_embed = PatchEmbedding3D(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, norm_layer=norm_layer
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer3D(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                patches_resolution[1] // (2 ** i_layer),
                                patches_resolution[2] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)
            
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


def swin3d_tiny(num_classes: int = 400, **kwargs):
    """Swin3D-T for video classification (Kinetics-400)"""
    model = SwinTransformer3D(
        img_size=(16, 224, 224), patch_size=(2, 4, 4),
        embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
        window_size=(8, 7, 7), num_classes=num_classes, **kwargs
    )
    return model


def swin3d_small(num_classes: int = 400, **kwargs):
    """Swin3D-S for video classification"""
    model = SwinTransformer3D(
        img_size=(16, 224, 224), patch_size=(2, 4, 4),
        embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24),
        window_size=(8, 7, 7), num_classes=num_classes, **kwargs
    )
    return model


def swin3d_base(num_classes: int = 400, **kwargs):
    """Swin3D-B for video classification"""
    model = SwinTransformer3D(
        img_size=(16, 224, 224), patch_size=(2, 4, 4),
        embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        window_size=(8, 7, 7), num_classes=num_classes, **kwargs
    )
    return model


def swin3d_large(num_classes: int = 400, **kwargs):
    """Swin3D-L for video classification"""
    model = SwinTransformer3D(
        img_size=(16, 224, 224), patch_size=(2, 4, 4),
        embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        window_size=(8, 7, 7), num_classes=num_classes, **kwargs
    )
    return model


# Medical imaging variants with different input sizes
def swin3d_medical_small(num_classes: int = 2, img_size: Tuple[int, int, int] = (64, 64, 64), **kwargs):
    """Swin3D for medical imaging (CT/MRI)"""
    model = SwinTransformer3D(
        img_size=img_size, patch_size=(4, 4, 4),
        in_channels=1,  # Single channel for medical imaging
        embed_dim=48, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
        window_size=(4, 4, 4), num_classes=num_classes, **kwargs
    )
    return model


if __name__ == "__main__":
    # Example 1: Video classification
    print("=== Video Classification Example ===")
    model_video = swin3d_tiny(num_classes=400)
    print(f"Model: Swin3D Transformer (Video)")
    print(f"Total parameters: {sum(p.numel() for p in model_video.parameters()) / 1e6:.2f}M")
    
    # Test with video input: batch_size=2, channels=3 (RGB), frames=16, height=224, width=224
    x_video = torch.randn(2, 3, 16, 224, 224)
    output_video = model_video(x_video)
    print(f"Input shape: {x_video.shape}")
    print(f"Output shape: {output_video.shape}")
    
    print("\n=== Medical Imaging Example ===")
    model_medical = swin3d_medical_small(num_classes=2, img_size=(64, 64, 64))
    print(f"Model: Swin3D Transformer (Medical)")
    print(f"Total parameters: {sum(p.numel() for p in model_medical.parameters()) / 1e6:.2f}M")
    
    # Test with medical volume: batch_size=2, channels=1 (grayscale), depth=64, height=64, width=64
    x_medical = torch.randn(2, 1, 64, 64, 64)
    output_medical = model_medical(x_medical)
    print(f"Input shape: {x_medical.shape}")
    print(f"Output shape: {output_medical.shape}")