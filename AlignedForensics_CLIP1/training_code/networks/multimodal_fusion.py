"""
多模态融合模块
结合VAE指纹特征和CLIP特征进行多模态融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, vae_dim, clip_dim, hidden_dim=512, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.vae_dim = vae_dim
        self.clip_dim = clip_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # 投影层
        self.vae_proj = nn.Linear(vae_dim, hidden_dim)
        self.clip_proj = nn.Linear(clip_dim, hidden_dim)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 层归一化和前馈网络
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, vae_features, clip_features):
        """
        Args:
            vae_features: VAE特征 [batch_size, vae_dim]
            clip_features: CLIP特征 [batch_size, clip_dim]
        
        Returns:
            fused_features: 融合后的特征 [batch_size, hidden_dim]
        """
        batch_size = vae_features.size(0)
        
        # 投影到相同维度
        vae_proj = self.vae_proj(vae_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        clip_proj = self.clip_proj(clip_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 拼接特征作为查询、键、值
        combined = torch.cat([vae_proj, clip_proj], dim=1)  # [batch_size, 2, hidden_dim]
        
        # 自注意力
        attn_output, _ = self.attention(combined, combined, combined)
        
        # 残差连接和层归一化
        attn_output = self.norm1(attn_output + combined)
        
        # 前馈网络
        ffn_output = self.ffn(attn_output)
        fused_features = self.norm2(ffn_output + attn_output)
        
        # 平均池化得到最终特征
        fused_features = fused_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        return fused_features

class MultimodalFusion(nn.Module):
    """多模态融合网络"""
    
    def __init__(self, vae_dim, clip_dim, hidden_dim=512, fusion_method="attention", 
                 num_classes=1, dropout=0.1):
        super(MultimodalFusion, self).__init__()
        self.vae_dim = vae_dim
        self.clip_dim = clip_dim
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        
        if fusion_method == "concat":
            # 简单拼接
            self.fusion_layer = nn.Sequential(
                nn.Linear(vae_dim + clip_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self._concat_in_dim = vae_dim + clip_dim
        elif fusion_method == "attention":
            # 跨模态注意力融合
            # 将不同维度映射到 hidden_dim 后做注意力
            self.vae_map = nn.Linear(vae_dim, hidden_dim)
            self.clip_map = nn.Linear(clip_dim, hidden_dim)
            self.fusion_layer = CrossModalAttention(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
            self._vae_in_dim = vae_dim
            self._clip_in_dim = clip_dim
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 特征提取器（用于返回中间特征）
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, vae_features, clip_features, return_features=False):
        """
        Args:
            vae_features: VAE特征 [batch_size, vae_dim]
            clip_features: CLIP特征 [batch_size, clip_dim]
            return_features: 是否返回中间特征
        
        Returns:
            output: 分类输出 [batch_size, num_classes]
            features: 中间特征（如果return_features=True）
        """
        # 多模态融合
        if self.fusion_method == "concat":
            # 若实际输入维度变化，则自适应重建首层线性
            in_dim = vae_features.size(1) + clip_features.size(1)
            if hasattr(self, "_concat_in_dim") and in_dim != self._concat_in_dim:
                device = next(self.parameters()).device
                # 仅重建第一层，保持后续结构
                new_first = nn.Linear(in_dim, self.hidden_dim).to(device)
                # 替换 Sequential 的第0层
                self.fusion_layer[0] = new_first
                self._concat_in_dim = in_dim
            fused_features = torch.cat([vae_features, clip_features], dim=1)
            fused_features = self.fusion_layer(fused_features)
        else:  # attention
            # 若实际输入维度变化，则自适应重建映射层
            vae_in = vae_features.size(1)
            clip_in = clip_features.size(1)
            device = next(self.parameters()).device
            if hasattr(self, "_vae_in_dim") and vae_in != self._vae_in_dim:
                self.vae_map = nn.Linear(vae_in, self.hidden_dim).to(device)
                self._vae_in_dim = vae_in
            if hasattr(self, "_clip_in_dim") and clip_in != self._clip_in_dim:
                self.clip_map = nn.Linear(clip_in, self.hidden_dim).to(device)
                self._clip_in_dim = clip_in
            vae_h = self.vae_map(vae_features)
            clip_h = self.clip_map(clip_features)
            fused_features = self.fusion_layer(vae_h, clip_h)
        
        # 特征提取
        extracted_features = self.feature_extractor(fused_features)
        
        # 分类
        output = self.classifier(extracted_features)
        
        if return_features:
            return output, {
                'vae_features': vae_features,
                'clip_features': clip_features,
                'fused_features': fused_features,
                'extracted_features': extracted_features
            }
        else:
            return output

class MultimodalDetector(nn.Module):
    """多模态假图像检测器"""
    
    def __init__(self, vae_backbone, clip_model, vae_dim, clip_dim, 
                 hidden_dim=512, fusion_method="attention", num_classes=1, 
                 freeze_clip=True, dropout=0.1):
        super(MultimodalDetector, self).__init__()
        self.vae_backbone = vae_backbone
        self.clip_model = clip_model
        self.fusion_method = fusion_method
        
        # 冻结CLIP模型（除了优化层）
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # 多模态融合网络
        self.fusion_net = MultimodalFusion(
            vae_dim=vae_dim,
            clip_dim=clip_dim,
            hidden_dim=hidden_dim,
            fusion_method=fusion_method,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # 特征投影层（此处保持恒等映射，真实维度调整在融合层中完成）
        self.vae_proj = nn.Identity()
        self.clip_proj = nn.Identity()
    
    def forward(self, images, captions=None, return_features=False):
        """
        Args:
            images: 输入图像 [batch_size, channels, height, width]
            captions: 图像描述（可选）[batch_size]
            return_features: 是否返回中间特征
        
        Returns:
            output: 分类输出
            features: 中间特征（如果return_features=True）
        """
        # VAE特征提取（输入已按训练管线做过 resnet 归一化）
        # 期望从骨干网拿到“特征”，而非最终logit。多数实现 forward(img, return_feats=True) 返回 (logit, feats)
        try:
            vae_out = self.vae_backbone(images, return_feats=True)
        except TypeError:
            # 如果不支持该关键字，则直接调用
            vae_out = self.vae_backbone(images)

        if isinstance(vae_out, tuple):
            # 约定 (logit, feats)
            if len(vae_out) == 2:
                vae_logits, vae_features = vae_out
            else:
                # 回退：取最后一个作为特征
                vae_features = vae_out[-1]
        else:
            # 只有一个输出，可能是特征也可能是logit；若形状为 [B,1] 之类，则无法作为特征使用
            vae_features = vae_out
        # 若是特征图，做全局平均池化到 [B, C]
        if isinstance(vae_features, torch.Tensor) and vae_features.dim() == 4:
            vae_features = F.adaptive_avg_pool2d(vae_features, (1, 1)).flatten(1)
        # 如果仍是 [B, 1]（极端退化为logit），则直接广播到隐藏维度前再映射
        if isinstance(vae_features, torch.Tensor) and vae_features.dim() == 2 and vae_features.size(1) == 1:
            # 将其重复到与 CLIP 特征相同的维度再由映射层适配
            vae_features = vae_features.repeat(1, 512)
        vae_features = self.vae_proj(vae_features)
        
        # CLIP特征提取
        # 注意：CLIP ViT-B/16 期望 224x224 尺寸，且使用自身的归一化。
        # 当前 images 使用的是 ResNet 归一化，需要先反归一化到 [0,1] 再按 CLIP 归一化后送入 encode_image。
        def _to_clip_input(x: torch.Tensor) -> torch.Tensor:
            # 反归一化 ResNet
            resnet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1,3,1,1)
            resnet_std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1,3,1,1)
            x = x * resnet_std + resnet_mean
            x = x.clamp(0, 1)
            # 调整到 224
            x = F.interpolate(x, size=224, mode='bicubic', align_corners=False)
            # 应用 CLIP 归一化
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device, dtype=x.dtype).view(1,3,1,1)
            clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device, dtype=x.dtype).view(1,3,1,1)
            x = (x - clip_mean) / clip_std
            return x

        clip_features = None
        with torch.no_grad() if not self.training else torch.enable_grad():
            clip_in = _to_clip_input(images)
            clip_features = self.clip_model.encode_image(clip_in)
            if captions is not None:
                try:
                    # 将字符串列表 tokenization 后再 encode_text
                    from networks.models.clip.clip import tokenize as clip_tokenize
                    if isinstance(captions, (list, tuple)):
                        tokens = clip_tokenize(list(captions)).to(clip_in.device)
                    else:
                        tokens = clip_tokenize([captions]).to(clip_in.device)
                    text_features = self.clip_model.encode_text(tokens)
                    # 简单融合：求和（也可改为 concat 再线性层）
                    if text_features.shape[0] == clip_features.shape[0]:
                        clip_features = clip_features + text_features
                except Exception:
                    # 若文本不可用或 tokenizer 缺失，则仅使用图像特征
                    pass
        clip_features = self.clip_proj(clip_features)
        
        # 多模态融合和分类
        if return_features:
            output, features = self.fusion_net(vae_features, clip_features, return_features=True)
            return output, features
        else:
            output = self.fusion_net(vae_features, clip_features, return_features=False)
            return output
    
    def get_attention_weights(self, images, captions=None):
        """获取注意力权重（用于可视化）"""
        # 这里可以实现注意力权重的提取
        # 具体实现取决于使用的融合方法
        pass

def create_multimodal_detector(vae_backbone, clip_model, vae_dim, clip_dim, 
                              fusion_method="attention", **kwargs):
    """创建多模态检测器的工厂函数"""
    return MultimodalDetector(
        vae_backbone=vae_backbone,
        clip_model=clip_model,
        vae_dim=vae_dim,
        clip_dim=clip_dim,
        fusion_method=fusion_method,
        **kwargs
    )

def add_multimodal_arguments(parser):
    """添加多模态相关参数"""
    parser.add_argument("--fusion_method", type=str, default="attention",
                       choices=["concat", "attention"],
                       help="Multimodal fusion method")
    parser.add_argument("--hidden_dim", type=int, default=512,
                       help="Hidden dimension for fusion")
    parser.add_argument("--freeze_clip", action="store_true", default=True,
                       help="Freeze CLIP model weights")
    parser.add_argument("--multimodal_dropout", type=float, default=0.1,
                       help="Dropout rate for multimodal layers")
    return parser

if __name__ == "__main__":
    # 示例用法
    import torch
    from transformers import CLIPModel
    
    # 模拟输入
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    captions = ["A photo of a cat", "A photo of a dog", "A photo of a bird", "A photo of a fish"]
    
    # 创建模型
    vae_backbone = torch.nn.Linear(1000, 512)  # 模拟VAE backbone
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    
    detector = create_multimodal_detector(
        vae_backbone=vae_backbone,
        clip_model=clip_model,
        vae_dim=512,
        clip_dim=512,
        fusion_method="attention"
    )
    
    # 前向传播
    output, features = detector(images, captions, return_features=True)
    print(f"Output shape: {output.shape}")
    print(f"Features keys: {features.keys()}")
