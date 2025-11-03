"""
可视化工具模块
提供注意力图、Grad-CAM等可解释性可视化功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from typing import Dict, List, Optional, Tuple
import seaborn as sns

class GradCAM:
    """Grad-CAM可视化"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # 找到目标层
        for name, module in model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                break
        
        if self.target_layer is None:
            raise ValueError(f"Target layer {target_layer_name} not found")
        
        # 注册钩子
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """保存激活值"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """保存梯度"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        """生成CAM图"""
        # 前向传播
        # 确保模型在训练模式以计算梯度
        self.model.train()
        model_output = self.model(input_tensor)
        if class_idx is None:
            class_idx = model_output.argmax(dim=1)
        
        # 反向传播
        self.model.zero_grad()
        # 确保输出是标量（对多类情况取最大，对二分类取对应类）
        if model_output.dim() > 1:
            if model_output.size(1) == 1:
                class_score = model_output.squeeze(1)
            else:
                class_score = model_output[:, class_idx].sum()  # 对batch求和得到标量
        else:
            class_score = model_output.sum()
        class_score.backward()
        
        # 计算权重
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # 生成CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # 归一化
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()

class AttentionVisualizer:
    """注意力可视化工具"""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
    
    def register_attention_hooks(self):
        """注册注意力钩子"""
        def attention_hook(module, input, output):
            if hasattr(module, 'attention_weights'):
                self.attention_weights.append(module.attention_weights.detach())
        
        # 为所有注意力层注册钩子
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                module.register_forward_hook(attention_hook)
    
    def visualize_attention_heads(self, attention_weights, save_path=None):
        """可视化注意力头"""
        if not attention_weights:
            print("No attention weights found")
            return
        
        # 取第一个注意力层的权重
        attn = attention_weights[0]  # [batch_size, num_heads, seq_len, seq_len]
        
        batch_size, num_heads, seq_len, _ = attn.shape
        
        fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()
        
        for head in range(num_heads):
            # 平均所有样本的注意力
            head_attn = attn[0, head].cpu().numpy()  # [seq_len, seq_len]
            
            im = axes[head].imshow(head_attn, cmap='Blues')
            axes[head].set_title(f'Head {head}')
            axes[head].set_xlabel('Key Position')
            axes[head].set_ylabel('Query Position')
            plt.colorbar(im, ax=axes[head])
        
        # 隐藏多余的子图
        for i in range(num_heads, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class FeatureVisualizer:
    """特征可视化工具"""
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
    
    def visualize_feature_maps(self, images, layer_name, save_path=None):
        """可视化特征图"""
        self.model.eval()
        
        # 注册钩子获取特征图
        feature_maps = []
        def hook(module, input, output):
            feature_maps.append(output.detach())
        
        # 找到目标层并注册钩子
        target_layer = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            print(f"Layer {layer_name} not found")
            return
        
        hook_handle = target_layer.register_forward_hook(hook)
        
        with torch.no_grad():
            _ = self.model(images.to(self.device))
        
        # 移除钩子
        hook_handle.remove()
        
        if not feature_maps:
            print("No feature maps captured")
            return
        
        # 可视化特征图
        feature_map = feature_maps[0][0]  # [C, H, W] - 第一个样本
        num_channels = min(64, feature_map.shape[0])  # 最多显示64个通道
        
        fig, axes = plt.subplots(8, 8, figsize=(16, 16))
        axes = axes.flatten()
        
        for i in range(num_channels):
            channel_map = feature_map[i].cpu().numpy()
            im = axes[i].imshow(channel_map, cmap='viridis')
            axes[i].set_title(f'Channel {i}')
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(num_channels, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_feature_distribution(self, features, labels, save_path=None):
        """可视化特征分布"""
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # 使用t-SNE降维，处理样本数不足的情况
        from sklearn.manifold import TSNE
        
        n_samples = features.shape[0]
        perplexity = min(30, max(5, n_samples // 4))  # 自适应perplexity
        
        if features.shape[1] > 2 and n_samples > perplexity:
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                features_2d = tsne.fit_transform(features)
            except Exception as e:
                print(f"t-SNE failed: {e}, using PCA instead")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                features_2d = pca.fit_transform(features)
        else:
            # 样本太少或维度已经很低，直接使用前两维
            if features.shape[1] >= 2:
                features_2d = features[:, :2]
            else:
                # 只有1维，复制一维
                features_2d = np.column_stack([features[:, 0], features[:, 0]])
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Feature Distribution (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class MultimodalVisualizer:
    """多模态可视化工具"""
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
    
    def visualize_multimodal_attention(self, images, captions=None, save_path=None):
        """可视化多模态注意力"""
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(self.model, 'fusion_net') and hasattr(self.model.fusion_net, 'fusion_layer'):
                if hasattr(self.model.fusion_net.fusion_layer, 'attention'):
                    # 获取注意力权重
                    vae_features = self.model.vae_backbone(images)
                    clip_features = self.model.clip_model.encode_image(images)
                    
                    # 通过融合层获取注意力
                    attention_output, attention_weights = self.model.fusion_net.fusion_layer.attention(
                        self.model.fusion_net.fusion_layer.vae_proj(vae_features).unsqueeze(1),
                        self.model.fusion_net.fusion_layer.clip_proj(clip_features).unsqueeze(1),
                        self.model.fusion_net.fusion_layer.clip_proj(clip_features).unsqueeze(1)
                    )
                    
                    # 可视化注意力权重
                    self._plot_attention_weights(attention_weights, save_path)
    
    def _plot_attention_weights(self, attention_weights, save_path=None):
        """绘制注意力权重"""
        # attention_weights: [batch_size, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
        if num_heads == 1:
            axes = [axes]
        
        for head in range(num_heads):
            attn = attention_weights[0, head].cpu().numpy()  # [seq_len, seq_len]
            
            im = axes[head].imshow(attn, cmap='Blues')
            axes[head].set_title(f'Head {head}')
            axes[head].set_xlabel('Key (VAE, CLIP)')
            axes[head].set_ylabel('Query (VAE, CLIP)')
            plt.colorbar(im, ax=axes[head])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interpretability_report(self, images, captions=None, output_dir="interpretability"):
        """创建可解释性报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Grad-CAM可视化
        try:
            # 尝试不同的层名，找到可用的卷积层
            layer_candidates = [
                'vae_backbone.layer4.2.conv3',
                'vae_backbone.layer4.1.conv3', 
                'vae_backbone.layer4.0.conv3',
                'vae_backbone.layer3.5.conv3',
                'vae_backbone.layer3.4.conv3'
            ]
            
            gradcam = None
            for layer_name in layer_candidates:
                try:
                    gradcam = GradCAM(self.model, layer_name)
                    cam = gradcam.generate_cam(images)
                    break
                except Exception:
                    continue
            
            if gradcam is not None:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                # 反归一化显示原图
                img_denorm = images[0].clone()
                img_denorm = img_denorm * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img_denorm = torch.clamp(img_denorm, 0, 1)
                plt.imshow(img_denorm.permute(1, 2, 0).cpu().numpy())
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(cam, cmap='jet')
                plt.title('Grad-CAM')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'gradcam.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("Grad-CAM visualization failed: No suitable layer found")
        except Exception as e:
            print(f"Grad-CAM visualization failed: {e}")
        
        # 2. 注意力可视化
        try:
            self.visualize_multimodal_attention(
                images, captions, 
                save_path=os.path.join(output_dir, 'attention_weights.png')
            )
        except Exception as e:
            print(f"Attention visualization failed: {e}")
        
        # 3. 特征分布可视化
        try:
            with torch.no_grad():
                _, features = self.model(images, captions, return_features=True)
            
            feature_vis = FeatureVisualizer(self.model, self.device)
            # 使用真实标签而不是模拟标签
            labels = torch.ones(len(images))  # 或者从数据中获取真实标签
            feature_vis.visualize_feature_distribution(
                features['fused_features'], 
                labels,
                save_path=os.path.join(output_dir, 'feature_distribution.png')
            )
        except Exception as e:
            print(f"Feature distribution visualization failed: {e}")

def add_visualization_arguments(parser):
    """添加可视化相关参数"""
    parser.add_argument("--enable_visualization", action="store_true",
                       help="Enable interpretability visualizations")
    parser.add_argument("--visualization_dir", type=str, default="interpretability",
                       help="Directory to save visualization results")
    parser.add_argument("--gradcam_layer", type=str, default="vae_backbone.layer4.2.conv3",
                       help="Layer name for Grad-CAM visualization")
    return parser

if __name__ == "__main__":
    # 示例用法
    import torch
    from transformers import CLIPModel
    
    # 模拟模型和输入
    model = torch.nn.Linear(1000, 2)  # 模拟模型
    images = torch.randn(4, 3, 224, 224)
    
    # 创建可视化工具
    visualizer = MultimodalVisualizer(model)
    
    # 生成可解释性报告
    visualizer.create_interpretability_report(images)
