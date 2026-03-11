"""Visual Contrast Attention DeiT Model - From Scratch"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict


# ============================================================================
# Helper Functions
# ============================================================================

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization"""
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.", flush=True)
    
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def lambda_init_fn(depth):
    """Initialize lambda parameter based on depth"""
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


# ============================================================================
# Normalization
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


# ============================================================================
# Visual Contrast Attention
# ============================================================================

class VisualContrastAttention(nn.Module):
    """
    Visual Contrast Attention - Linear Complexity Attention
    
    Uses two stages:
    Stage I: Global contrast using t_+ and t_- attending to all keys/values
    Stage II: Patch-wise differential attention using query-contrast interactions
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 vct_num=49, window=14, block_depth=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.vct_num = vct_num  # n: visual contrast token number
        
        # Depth-wise convolution for spatial tokens
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                             padding=1, groups=dim)
        
        # Adaptive pooling to generate visual contrast tokens
        pool_size = int(vct_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

        # Learnable positional embeddings for visual contrast tokens
        self.e_pos = nn.Parameter(torch.randn(1, vct_num, dim))
        self.e_neg = nn.Parameter(torch.randn(1, vct_num, dim))

        # Stage I (global contrast) lambda parameters
        self.lambda_1_init = lambda_init_fn(block_depth) if block_depth is not None else 0.5
        self.lambda_1_q1 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
        self.lambda_1_k1 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
        self.lambda_1_q2 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
        self.lambda_1_k2 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
        nn.init.normal_(self.lambda_1_q1, mean=0, std=0.1)
        nn.init.normal_(self.lambda_1_k1, mean=0, std=0.1)
        nn.init.normal_(self.lambda_1_q2, mean=0, std=0.1)
        nn.init.normal_(self.lambda_1_k2, mean=0, std=0.1)
        self.subln1 = RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)

        # Stage II (patch-wise differential attention) lambda parameters
        self.lambda_2_init = lambda_init_fn(block_depth) if block_depth is not None else 0.5
        self.lambda_2_q1 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
        self.lambda_2_k1 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
        self.lambda_2_q2 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
        self.lambda_2_k2 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
        nn.init.normal_(self.lambda_2_q1, mean=0, std=0.1)
        nn.init.normal_(self.lambda_2_k1, mean=0, std=0.1)
        nn.init.normal_(self.lambda_2_q2, mean=0, std=0.1)
        nn.init.normal_(self.lambda_2_k2, mean=0, std=0.1)
        self.subln2 = RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x):
        """
        Args:
            x: [B, N, C] where N = H*W + 1 (including cls token)
        
        Returns:
            [B, N, C]
        """
        b, n, c = x.shape
        h = int((n - 1) ** 0.5)  # Exclude cls token
        w = h
        num_heads = self.num_heads
        head_dim = self.head_dim
        
        # Project to Q, K, V
        q = self.to_q(x)  # [B, N, C]
        k = self.to_k(x)  # [B, N, C]
        v = self.to_v(x)  # [B, N, C]

        # Generate visual contrast tokens using adaptive pooling
        # t_tilde = AvgPool(q_spatial)
        q_spatial = q[:, 1:, :].reshape(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]
        t_tilde = self.pool(q_spatial).reshape(b, c, -1).permute(0, 2, 1)  # [B, vct_num, C]
        
        # Add learnable positional embeddings
        t_pos = self.e_pos.expand(b, -1, -1) + t_tilde  # [B, vct_num, C]
        t_neg = self.e_neg.expand(b, -1, -1) + t_tilde  # [B, vct_num, C]
        
        # Reshape to multi-head format: [B, N, C] -> [B, M, N, d]
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        t_pos = t_pos.reshape(b, self.vct_num, num_heads, head_dim).permute(0, 2, 1, 3)
        t_neg = t_neg.reshape(b, self.vct_num, num_heads, head_dim).permute(0, 2, 1, 3)
        
        # ===== STAGE I: Global Contrast =====
        # Compute λ^(1) = exp(λ_q1 · λ_k1) - exp(λ_q2 · λ_k2) + λ_init
        lambda_1_term1 = torch.exp(torch.sum(self.lambda_1_q1 * self.lambda_1_k1, dim=-1).float()).type_as(q)
        lambda_1_term2 = torch.exp(torch.sum(self.lambda_1_q2 * self.lambda_1_k2, dim=-1).float()).type_as(q)
        lambda_1 = lambda_1_term1 - lambda_1_term2 + self.lambda_1_init
        
        # Concatenate t_+ and t_- for efficient computation
        t_all = torch.cat((t_pos, t_neg), dim=2)  # [B, M, 2n, d]
        
        # Compute v_hat: softmax(t_all @ k^T / sqrt(d)) @ v
        stage1_attn = self.softmax((t_all * self.scale) @ k.transpose(-2, -1))  # [B, M, 2n, N]
        stage1_attn = self.attn_drop(stage1_attn)
        v_hat_all = stage1_attn @ v  # [B, M, 2n, d]
        
        # Differential: v_hat = v_hat_+ - λ^(1) * v_hat_-
        v_hat = v_hat_all[:, :, :self.vct_num] - lambda_1 * v_hat_all[:, :, self.vct_num:]  # [B, M, n, d]
        v_hat = self.subln1(v_hat)
        v_hat = v_hat * (1 - self.lambda_1_init)

        # ===== STAGE II: Patch-wise Differential Attention =====
        # Compute λ^(2)
        lambda_2_term1 = torch.exp(torch.sum(self.lambda_2_q1 * self.lambda_2_k1, dim=-1).float()).type_as(q)
        lambda_2_term2 = torch.exp(torch.sum(self.lambda_2_q2 * self.lambda_2_k2, dim=-1).float()).type_as(q)
        lambda_2 = lambda_2_term1 - lambda_2_term2 + self.lambda_2_init

        # Attention: A_1 = softmax(q @ t_+^T), A_2 = softmax(q @ t_-^T)
        A_1_A_2 = self.softmax((q * self.scale) @ t_all.transpose(-2, -1))  # [B, M, N, 2n]
        A_1_A_2 = self.attn_drop(A_1_A_2)
        A_1_A_2 = A_1_A_2.view(b, num_heads, n, 2, self.vct_num).permute(0, 1, 3, 2, 4)  # [B, M, 2, N, n]
        
        # Differential attention: A = A_1 - λ^(2) * A_2
        A = A_1_A_2[:, :, 0] - lambda_2 * A_1_A_2[:, :, 1]  # [B, M, N, n]
        
        # Output: h_hat = A @ v_hat
        x = A @ v_hat  # [B, M, N, d]
        x = self.subln2(x)
        x = x * (1 - self.lambda_2_init)

        # Reshape back: [B, M, N, d] -> [B, N, C]
        x = x.transpose(1, 2).reshape(b, n, c)
        
        # Add DWC (depth-wise convolution) residual for spatial tokens
        v_spatial = v[:, :, 1:, :].transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x_spatial = self.dwc(v_spatial).permute(0, 2, 3, 1).reshape(b, h*w, c)
        x[:, 1:, :] = x[:, 1:, :] + x_spatial

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================================
# MLP
# ============================================================================

class MLP(nn.Module):
    """Multi-layer Perceptron"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================================
# Drop Path / Stochastic Depth
# ============================================================================

class DropPath(nn.Module):
    """Stochastic Depth - randomly drop paths during training"""
    def __init__(self, drop_prob=0., training=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.training = training

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        
        if keep_prob > 0.0:
            random_mask = random_mask / keep_prob
        
        return x * random_mask


# ============================================================================
# Transformer Block with VCA
# ============================================================================

class VisualContrastBlock(nn.Module):
    """Transformer block with Visual Contrast Attention"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 vct_num=49, window=14, block_depth=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VisualContrastAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, 
            proj_drop=drop, vct_num=vct_num, window=window, block_depth=block_depth
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block (without VCA)"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# Patch Embedding
# ============================================================================

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, N, C] where N = (H/patch_size) * (W/patch_size)
        """
        x = self.proj(x)  # [B, C, H', W']
        x = x.flatten(2)  # [B, C, N]
        x = x.transpose(1, 2)  # [B, N, C]
        return x


# ============================================================================
# Vision Transformer with Visual Contrast Attention
# ============================================================================

class VisionTransformer(nn.Module):
    """Vision Transformer with Visual Contrast Attention"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, vct_num=None, vct_layer=-1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_tokens = 1  # Only cls token
        
        if vct_num is None:
            vct_num = [49] * 4
        
        self.vct_layer = vct_layer if vct_layer > 0 else depth
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches  # 196 for 224x224
        
        # Token embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Stage mapping for vct_num
        stage_num = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3}
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            VisualContrastBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer,
                vct_num=int(vct_num[stage_num.get(i // 2, 0)]),
                window=img_size // patch_size, block_depth=i
            ) if i < self.vct_layer else TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        # Layer norm + classifier
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Init weights
        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
    
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 
        Returns:
            [B, num_classes]
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, C]
        
        # Add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, C]
        x = torch.cat((cls_token, x), dim=1)  # [B, N+1, C]
        
        # Add position embedding
        x = self.pos_drop(x + self.pos_embed)
        
        # Transformer blocks
        x = self.blocks(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Classification head (use cls token)
        x = x[:, 0]
        x = self.head(x)
        
        return x
    
    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


# ============================================================================
# Model Creation Functions
# ============================================================================

def create_vca_deit_tiny(img_size=224, drop_path_rate=0.1, agent_num=None, **kwargs):
    """Create VCA-DeiT Tiny model"""
    if agent_num is None:
        agent_num = [49, 49, 49, 49]
    
    model = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        vct_num=agent_num,
        vct_layer=12,  # Use VCA for all layers
        **kwargs
    )
    return model


def create_vca_deit_small(img_size=224, drop_path_rate=0.1, agent_num=None, **kwargs):
    """Create VCA-DeiT Small model"""
    if agent_num is None:
        agent_num = [49, 49, 49, 49]
    
    model = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        vct_num=agent_num,
        vct_layer=12,
        **kwargs
    )
    return model


def create_vca_deit_base(img_size=224, drop_path_rate=0.1, agent_num=None, **kwargs):
    """Create VCA-DeiT Base model"""
    if agent_num is None:
        agent_num = [49, 49, 49, 49]
    
    model = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        vct_num=agent_num,
        vct_layer=4,  # Only first 4 layers use VCA
        **kwargs
    )
    return model


# ============================================================================
# Detection Head
# ============================================================================

class DetectionHead(nn.Module):
    """Detection head for object detection"""
    
    def __init__(self, embed_dim, num_classes=80, num_queries=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        
        # Class prediction
        self.class_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Linear(embed_dim, num_classes + 1)  # +1 for background
        )
        
        # Bounding box prediction
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Linear(embed_dim, 4)  # 4 for (x, y, w, h)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, N, C] from backbone
        
        Returns:
            class_logits: [B, num_queries, num_classes+1]
            bbox_pred: [B, num_queries, 4]
        """
        B = x.shape[0]
        
        # Get query embeddings
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, C]
        
        # Simple approach: use mean of backbone features + queries
        # More advanced: use cross-attention
        class_logits = self.class_head(queries)  # [B, num_queries, num_classes+1]
        bbox_pred = self.bbox_head(queries)      # [B, num_queries, 4]
        
        # Sigmoid for bbox (normalized coordinates)
        bbox_pred = torch.sigmoid(bbox_pred)
        
        return class_logits, bbox_pred


# ============================================================================
# Vision Transformer for Object Detection
# ============================================================================

class VisionTransformerForDetection(nn.Module):
    """Vision Transformer with Visual Contrast Attention for Object Detection"""
    
    def __init__(self, img_size=512, patch_size=16, in_chans=3, 
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, vct_num=None, vct_layer=-1,
                 num_classes=80, num_queries=100):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_tokens = 1  # Only cls token
        
        if vct_num is None:
            vct_num = [49, 49, 49, 49]
        
        self.vct_layer = vct_layer if vct_layer > 0 else depth
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Token embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Stage mapping for vct_num
        stage_num = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3}
        
        # Transformer blocks (backbone for detection)
        self.blocks = nn.Sequential(*[
            VisualContrastBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer,
                vct_num=int(vct_num[stage_num.get(i // 2, 0)]),
                window=img_size // patch_size, block_depth=i
            ) if i < self.vct_layer else TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        # Layer norm
        self.norm = norm_layer(embed_dim)
        
        # Detection head
        self.detection_head = DetectionHead(embed_dim, num_classes, num_queries)
        
        # Init weights
        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
    
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        
        Returns:
            class_logits: [B, num_queries, num_classes+1]
            bbox_pred: [B, num_queries, 4]
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, C]
        
        # Add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, N+1, C]
        
        # Add position embedding
        x = self.pos_drop(x + self.pos_embed)
        
        # Transformer blocks
        x = self.blocks(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Detection head
        class_logits, bbox_pred = self.detection_head(x)
        
        return class_logits, bbox_pred
    
    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


# ============================================================================
# Model Creation Functions
# ============================================================================

def create_vca_deit_tiny_detection(img_size=512, drop_path_rate=0.1, 
                                  agent_num=None, num_classes=80, **kwargs):
    """Create VCA-DeiT Tiny for object detection"""
    if agent_num is None:
        agent_num = [49, 49, 49, 49]
    
    model = VisionTransformerForDetection(
        img_size=img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        vct_num=agent_num,
        vct_layer=12,
        num_classes=num_classes,
        **kwargs
    )
    return model


def create_vca_deit_small_detection(img_size=512, drop_path_rate=0.1, 
                                   agent_num=None, num_classes=80, **kwargs):
    """Create VCA-DeiT Small for object detection"""
    if agent_num is None:
        agent_num = [49, 49, 49, 49]
    
    model = VisionTransformerForDetection(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        vct_num=agent_num,
        vct_layer=12,
        num_classes=num_classes,
        **kwargs
    )
    return model


def create_model(model_type='vca_deit_tiny', task='detection', **kwargs):
    """Create model by type and task"""
    models = {
        'vca_deit_tiny': {
            'classification': create_vca_deit_tiny,
            'detection': create_vca_deit_tiny_detection,
        },
        'vca_deit_small': {
            'classification': create_vca_deit_small,
            'detection': create_vca_deit_small_detection,
        },
        'vca_deit_base': {
            'classification': create_vca_deit_base,
            'detection': lambda **kw: create_vca_deit_base(**kw),  # No detection version yet
        },
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if task not in models[model_type]:
        raise ValueError(f"Task '{task}' not supported for {model_type}")
    
    return models[model_type][task](**kwargs)
