"""
特征优化模块
用于优化CLIP特征，去除有害维度并选择重要的注意力头
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os

class FeatureOptimizer:
    """特征优化器"""
    
    def __init__(self, clip_model, device="cuda"):
        self.clip_model = clip_model
        self.device = device
        self.feature_importance = None
        self.attention_head_importance = None
    
    def prune_embedding(self, clip_features, labels=None, method="variance", 
                       threshold=0.1, keep_ratio=0.8):
        """
        分析CLIP图像嵌入维度并移除有害或冗余维度
        
        Args:
            clip_features: CLIP图像特征 [batch_size, feature_dim]
            labels: 真实/假图像标签 [batch_size]
            method: 选择方法 ("variance", "mutual_info", "pca")
            threshold: 方差阈值
            keep_ratio: 保留特征的比例
        
        Returns:
            pruned_features: 优化后的特征
            feature_mask: 特征掩码
        """
        if isinstance(clip_features, torch.Tensor):
            clip_features = clip_features.cpu().numpy()
        
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        if method == "variance":
            # 基于方差的特征选择
            feature_vars = np.var(clip_features, axis=0)
            if threshold is not None:
                feature_mask = feature_vars > threshold
            else:
                # 保留方差最大的特征
                n_keep = int(len(feature_vars) * keep_ratio)
                top_indices = np.argsort(feature_vars)[-n_keep:]
                feature_mask = np.zeros(len(feature_vars), dtype=bool)
                feature_mask[top_indices] = True
        
        elif method == "mutual_info" and labels is not None:
            # 基于互信息的特征选择
            mi_scores = mutual_info_classif(clip_features, labels)
            if threshold is not None:
                feature_mask = mi_scores > threshold
            else:
                n_keep = int(len(mi_scores) * keep_ratio)
                top_indices = np.argsort(mi_scores)[-n_keep:]
                feature_mask = np.zeros(len(mi_scores), dtype=bool)
                feature_mask[top_indices] = True
        
        elif method == "pca":
            # 使用PCA降维
            pca = PCA(n_components=int(len(clip_features[0]) * keep_ratio))
            pruned_features = pca.fit_transform(clip_features)
            return torch.tensor(pruned_features).to(self.device), None
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # 应用特征掩码
        pruned_features = clip_features[:, feature_mask]
        
        return torch.tensor(pruned_features).to(self.device), feature_mask
    
    def select_attention_heads(self, model, head_importance_scores=None, 
                             keep_ratio=0.7, fine_tune_heads=True):
        """
        评估CLIP ViT编码器中的注意力头重要性并保留最有信息的头
        
        Args:
            model: CLIP模型
            head_importance_scores: 预计算的注意力头重要性分数
            keep_ratio: 保留注意力头的比例
            fine_tune_heads: 是否只微调选中的头
        
        Returns:
            selected_heads: 选中的注意力头索引
            head_mask: 注意力头掩码
        """
        if head_importance_scores is None:
            # 计算注意力头重要性（基于梯度的近似）
            head_importance_scores = self._compute_head_importance(model)
        
        # 选择最重要的头
        n_heads = len(head_importance_scores)
        n_keep = int(n_heads * keep_ratio)
        top_heads = np.argsort(head_importance_scores)[-n_keep:]
        
        head_mask = np.zeros(n_heads, dtype=bool)
        head_mask[top_heads] = True
        
        if fine_tune_heads:
            # 冻结未选中的注意力头
            self._freeze_unselected_heads(model, head_mask)
        
        return top_heads, head_mask
    
    def _compute_head_importance(self, model, num_samples=100):
        """计算注意力头重要性分数"""
        model.eval()
        head_importance = []
        
        # 获取ViT编码器
        if hasattr(model, 'visual'):
            vit = model.visual
        else:
            vit = model
        
        # 找到transformer层
        transformer_layers = []
        for name, module in vit.named_modules():
            if 'transformer.resblocks' in name and 'attn' in name:
                transformer_layers.append(module)
        
        if not transformer_layers:
            print("Warning: Could not find transformer layers for head importance computation")
            return np.ones(12)  # 默认值
        
        # 计算每个头的梯度范数作为重要性指标
        for layer in transformer_layers:
            if hasattr(layer, 'in_proj_weight'):
                # Multi-head attention层
                in_proj_weight = layer.in_proj_weight
                num_heads = 12  # CLIP ViT-B/16默认12个头
                head_dim = in_proj_weight.shape[0] // (3 * num_heads)
                
                for head in range(num_heads):
                    start_idx = head * head_dim
                    end_idx = (head + 1) * head_dim
                    head_weight = in_proj_weight[start_idx:end_idx]
                    importance = torch.norm(head_weight).item()
                    head_importance.append(importance)
        
        return np.array(head_importance) if head_importance else np.ones(12)
    
    def _freeze_unselected_heads(self, model, head_mask):
        """冻结未选中的注意力头"""
        def freeze_heads(module, head_mask, head_idx=0):
            if hasattr(module, 'in_proj_weight'):
                # 冻结未选中的头的参数
                for param in module.parameters():
                    if not head_mask[head_idx % len(head_mask)]:
                        param.requires_grad = False
                return head_idx + 1
            return head_idx
        
        # 递归冻结注意力头
        head_idx = 0
        for name, module in model.named_modules():
            if 'attn' in name:
                head_idx = freeze_heads(module, head_mask, head_idx)
    
    def visualize_feature_importance(self, feature_importance, save_path=None):
        """可视化特征重要性"""
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(feature_importance)
        plt.title('Feature Importance Scores')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        
        plt.subplot(1, 2, 2)
        plt.hist(feature_importance, bins=50)
        plt.title('Feature Importance Distribution')
        plt.xlabel('Importance')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def optimize_features(self, clip_features, labels=None, 
                         embedding_method="variance", 
                         attention_method=True,
                         **kwargs):
        """
        综合特征优化
        
        Args:
            clip_features: CLIP特征
            labels: 标签
            embedding_method: 嵌入优化方法
            attention_method: 是否进行注意力头优化
        
        Returns:
            optimized_features: 优化后的特征
            optimization_info: 优化信息字典
        """
        optimization_info = {}
        
        # 1. 嵌入维度优化
        if embedding_method:
            pruned_features, feature_mask = self.prune_embedding(
                clip_features, labels, method=embedding_method, **kwargs
            )
            optimization_info['feature_mask'] = feature_mask
            optimization_info['n_features_kept'] = np.sum(feature_mask) if feature_mask is not None else len(clip_features[0])
        else:
            pruned_features = clip_features
        
        # 2. 注意力头优化
        if attention_method:
            selected_heads, head_mask = self.select_attention_heads(
                self.clip_model, keep_ratio=kwargs.get('keep_ratio', 0.7)
            )
            optimization_info['selected_heads'] = selected_heads
            optimization_info['head_mask'] = head_mask
        
        return pruned_features, optimization_info

def add_feature_optimization_arguments(parser):
    """添加特征优化相关参数"""
    parser.add_argument("--enable_feature_optimization", action="store_true",
                       help="Enable CLIP feature optimization")
    parser.add_argument("--embedding_method", type=str, default="variance",
                       choices=["variance", "mutual_info", "pca"],
                       help="Method for embedding optimization")
    parser.add_argument("--keep_ratio", type=float, default=0.8,
                       help="Ratio of features/heads to keep")
    parser.add_argument("--optimization_threshold", type=float, default=0.1,
                       help="Threshold for feature selection")
    return parser

if __name__ == "__main__":
    # 示例用法
    import torch
    from transformers import CLIPModel, CLIPProcessor
    
    # 加载CLIP模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    # 创建特征优化器
    optimizer = FeatureOptimizer(model)
    
    # 模拟特征和标签
    batch_size = 32
    feature_dim = 512
    clip_features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, 2, (batch_size,))
    
    # 优化特征
    optimized_features, info = optimizer.optimize_features(
        clip_features, labels, embedding_method="variance"
    )
    
    print(f"Original features: {clip_features.shape}")
    print(f"Optimized features: {optimized_features.shape}")
    print(f"Features kept: {info['n_features_kept']}")
