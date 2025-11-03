"""
多模态训练模块
扩展原有的训练模型以支持CLIP多模态特征
"""

import os
import sys

import torch
import torch.nn as nn
import numpy as np
import tqdm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from typing import Dict, Optional, Tuple, List
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from .training import TrainingModel
from training_code.networks.multimodal_fusion import MultimodalDetector, create_multimodal_detector
from training_code.networks.models.clip_models import CLIPModel
from .feature_optimization import FeatureOptimizer
from .captioning import ImageCaptioner
from .visualization import MultimodalVisualizer

class MultimodalTrainingModel(TrainingModel):
    """多模态训练模型"""
    
    def __init__(self, opt, subdir='.'):
        # 初始化基础训练模型
        super().__init__(opt, subdir)
        
        # 多模态相关参数
        self.enable_multimodal = getattr(opt, 'enable_multimodal', False)
        self.enable_captioning = getattr(opt, 'enable_captioning', False)
        self.enable_feature_optimization = getattr(opt, 'enable_feature_optimization', False)
        self.fusion_method = getattr(opt, 'fusion_method', 'attention')
        self.hidden_dim = getattr(opt, 'hidden_dim', 512)
        
        if self.enable_multimodal:
            self._setup_multimodal_components()
    
    def _setup_multimodal_components(self):
        """设置多模态组件"""
        # 1. 加载CLIP模型
        try:
            self.clip_model = CLIPModel("ViT-B/16", num_classes=1)
            #self.clip_model = CLIPModel("D:\Model\AlignedForensics_CLIP\training_code\ViT-B-16.pt", num_classes=1)
            self.clip_model.to(self.device)
        except Exception as e:
            print(f"Warning: CLIP model loading failed: {e}")
            print("Falling back to basic CLIP integration")
            self.clip_model = None
        
        # 2. 创建多模态检测器
        vae_dim = 2048  # ResNet-50最后一层特征维度
        clip_dim = 512  # CLIP ViT-B/16特征维度
        
        self.multimodal_detector = create_multimodal_detector(
            vae_backbone=self.model,
            clip_model=self.clip_model.model,
            vae_dim=vae_dim,
            clip_dim=clip_dim,
            hidden_dim=self.hidden_dim,
            fusion_method=self.fusion_method,
            freeze_clip=getattr(self.opt, 'freeze_clip', True)
        )
        # 确保整套多模态检测器在同一设备
        self.multimodal_detector = self.multimodal_detector.to(self.device)
        
        # 3. 特征优化器
        if self.enable_feature_optimization:
            self.feature_optimizer = FeatureOptimizer(
                self.clip_model.model, self.device
            )
        
        # 4. 图像描述生成器
        if self.enable_captioning:
            self.captioner = ImageCaptioner(
                model_type=getattr(self.opt, 'caption_model', 'blip'),
                device=self.device
            )
        
        # 5. 可视化工具
        if getattr(self.opt, 'enable_visualization', False):
            self.visualizer = MultimodalVisualizer(
                self.multimodal_detector, self.device
            )
        
        # 6. 更新优化器以包含多模态参数
        self._update_optimizer()
    
    def _update_optimizer(self):
        """更新优化器以包含多模态参数"""
        if not self.enable_multimodal:
            return
        
        # 仅使用多模态检测器中可训练参数，避免重复加入同一骨干网络导致重复参数警告
        trainable_params = filter(lambda p: p.requires_grad, self.multimodal_detector.parameters())
        
        if self.opt.optim == "adam":
            self.optimizer = torch.optim.Adam(
                trainable_params, 
                lr=self.opt.lr, 
                betas=(self.opt.beta1, 0.999), 
                weight_decay=self.opt.weight_decay
            )
        elif self.opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                trainable_params, 
                lr=self.opt.lr, 
                momentum=0.0, 
                weight_decay=self.opt.weight_decay
            )
    
    def train_on_batch(self, data):
        """多模态训练批次"""
        if not self.enable_multimodal:
            return super().train_on_batch(data)
        
        self.model.train()
        self.multimodal_detector.train()
        
        # 准备数据
        if self.opt.batched_syncing:
            rdata, fdata = data[0], data[1]
            images = torch.cat((rdata['img'], fdata['img']), dim=0).to(self.device)
            labels = torch.cat((rdata['target'], fdata['target']), dim=0).to(self.device).float()
            captions = None
            
            # 生成图像描述（如果需要）
            if self.enable_captioning:
                captions = self._generate_captions(images)
        else:
            images = data['img'].to(self.device)
            labels = data['target'].to(self.device).float()
            captions = None
            
            if self.enable_captioning:
                captions = self._generate_captions(images)
        
        # 前向传播
        if self.enable_feature_optimization:
            # 使用特征优化
            output, features = self.multimodal_detector(images, captions, return_features=True)
            
            # 优化CLIP特征
            clip_features = features['clip_features']
            optimized_clip_features, _ = self.feature_optimizer.optimize_features(
                clip_features, labels, 
                embedding_method=getattr(self.opt, 'embedding_method', 'variance'),
                keep_ratio=getattr(self.opt, 'keep_ratio', 0.8)
            )
            
            # 重新融合特征
            vae_features = features['vae_features']
            output = self.multimodal_detector.fusion_net(optimized_clip_features, vae_features)
        else:
            output = self.multimodal_detector(images, captions)
        
        # 计算损失
        if len(output.shape) == 4:
            ss = output.shape
            loss = self.loss_fn(
                output,
                labels[:, None, None, None].repeat(
                    (1, int(ss[1]), int(ss[2]), int(ss[3]))
                ),
            )
        else:
            loss = self.loss_fn(output.squeeze(1), labels)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Stay-Positive Update (ICML)
        if self.opt.stay_positive == 'clamp':
            with torch.no_grad():
                if hasattr(self.multimodal_detector.fusion_net, 'classifier'):
                    for layer in self.multimodal_detector.fusion_net.classifier:
                        if isinstance(layer, nn.Linear):
                            layer.weight.data.clamp_(min=0)
        
        return loss.cpu()
    
    def _generate_captions(self, images):
        """生成图像描述"""
        if not self.enable_captioning:
            return None
        
        captions = []
        for i in range(images.size(0)):
            try:
                # 将tensor转换为PIL图像
                img_tensor = images[i]
                img_array = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_array = (img_array * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_array)
                
                caption = self.captioner.generate_caption(img_pil)
                captions.append(caption)
            except Exception as e:
                print(f"Caption generation failed for image {i}: {e}")
                captions.append("")
        
        return captions
    
    def predict(self, data_loader):
        """多模态预测"""
        if not self.enable_multimodal:
            return super().predict(data_loader)
        
        self.multimodal_detector.eval()
        
        with torch.no_grad():
            y_true, y_pred, y_path = [], [], []
            
            for data in tqdm.tqdm(data_loader):
                images = data['img'].to(self.device)
                labels = data['target'].cpu().numpy()
                paths = list(data['path'])
                
                # 生成描述（如果需要）
                captions = None
                if self.enable_captioning:
                    captions = self._generate_captions(images)
                
                # 预测
                output = self.multimodal_detector(images, captions)
                output = output.cpu().numpy()[:, -1]
                
                y_pred.extend(output.tolist())
                y_true.extend(labels.tolist())
                y_path.extend(paths)
        
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return y_true, y_pred, y_path
    
    def evaluate_multimodal_components(self, data_loader):
        """评估多模态组件的单独性能"""
        if not self.enable_multimodal:
            return {}
        
        self.multimodal_detector.eval()
        
        vae_accuracies, clip_accuracies, fused_accuracies = [], [], []
        
        with torch.no_grad():
            for data in tqdm.tqdm(data_loader):
                images = data['img'].to(self.device)
                labels = data['target'].to(self.device).float()
                
                # 生成描述
                captions = None
                if self.enable_captioning:
                    captions = self._generate_captions(images)
                
                # 获取各组件输出
                output, features = self.multimodal_detector(images, captions, return_features=True)

                vae_features = features['vae_features']
                clip_features = features['clip_features']
                fusion = self.multimodal_detector.fusion_net

                # 使用与当前融合方法一致的映射来评估单独模态
                if self.fusion_method == 'concat':
                    # 仅VAE：拼接 VAE 与 全零 CLIP
                    zeros_clip = torch.zeros_like(clip_features)
                    vae_only_fused = fusion.fusion_layer(torch.cat([vae_features, zeros_clip], dim=1))
                    vae_only_out = fusion.classifier(fusion.feature_extractor(vae_only_fused))
                    vae_pred = (vae_only_out.squeeze(1) > 0).float()
                    vae_acc = (vae_pred == labels).float().mean().item()
                    vae_accuracies.append(vae_acc)

                    # 仅CLIP：拼接 全零 VAE 与 CLIP
                    zeros_vae = torch.zeros_like(vae_features)
                    clip_only_fused = fusion.fusion_layer(torch.cat([zeros_vae, clip_features], dim=1))
                    clip_only_out = fusion.classifier(fusion.feature_extractor(clip_only_fused))
                    clip_pred = (clip_only_out.squeeze(1) > 0).float()
                    clip_acc = (clip_pred == labels).float().mean().item()
                    clip_accuracies.append(clip_acc)

                else:  # attention
                    # 将两个模态各自映射到 hidden_dim
                    vae_h = fusion.vae_map(vae_features)
                    clip_h = fusion.clip_map(clip_features)

                    # 仅VAE：CLIP 置零
                    clip_zero = torch.zeros_like(clip_h)
                    vae_only_fused = fusion.fusion_layer(vae_h, clip_zero)
                    vae_only_out = fusion.classifier(fusion.feature_extractor(vae_only_fused))
                    vae_pred = (vae_only_out.squeeze(1) > 0).float()
                    vae_acc = (vae_pred == labels).float().mean().item()
                    vae_accuracies.append(vae_acc)

                    # 仅CLIP：VAE 置零
                    vae_zero = torch.zeros_like(vae_h)
                    clip_only_fused = fusion.fusion_layer(vae_zero, clip_h)
                    clip_only_out = fusion.classifier(fusion.feature_extractor(clip_only_fused))
                    clip_pred = (clip_only_out.squeeze(1) > 0).float()
                    clip_acc = (clip_pred == labels).float().mean().item()
                    clip_accuracies.append(clip_acc)

                # 融合特征分类
                fused_pred = (output.squeeze(1) > 0).float()
                fused_acc = (fused_pred == labels).float().mean().item()
                fused_accuracies.append(fused_acc)
        
        return {
            'vae_accuracy': np.mean(vae_accuracies),
            'clip_accuracy': np.mean(clip_accuracies),
            'fused_accuracy': np.mean(fused_accuracies)
        }
    
    def create_interpretability_visualization(self, data_loader, output_dir="interpretability"):
        """创建可解释性可视化"""
        if not self.enable_multimodal or not hasattr(self, 'visualizer'):
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取一个批次的数据进行可视化
        for data in data_loader:
            images = data['img'].to(self.device)
            captions = None
            
            if self.enable_captioning:
                captions = self._generate_captions(images)
            
            # 创建可视化报告
            self.visualizer.create_interpretability_report(
                images, captions, output_dir
            )
            break  # 只可视化第一个批次

def add_multimodal_training_arguments(parser):
    """添加多模态训练参数"""
    # 基础多模态参数
    parser.add_argument("--enable_multimodal", action="store_true",
                       help="Enable multimodal training with CLIP")
    parser.add_argument("--fusion_method", type=str, default="attention",
                       choices=["concat", "attention"],
                       help="Multimodal fusion method")
    parser.add_argument("--hidden_dim", type=int, default=512,
                       help="Hidden dimension for multimodal fusion")
    parser.add_argument("--freeze_clip", action="store_true", default=True,
                       help="Freeze CLIP model weights")
    
    # 图像描述参数
    from .captioning import add_captioning_arguments
    parser = add_captioning_arguments(parser)
    
    # 特征优化参数
    from .feature_optimization import add_feature_optimization_arguments
    parser = add_feature_optimization_arguments(parser)
    
    # 可视化参数
    from .visualization import add_visualization_arguments
    parser = add_visualization_arguments(parser)
    
    return parser

if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser()
    parser = add_multimodal_training_arguments(parser)
    args = parser.parse_args()
    
    # 创建多模态训练模型
    model = MultimodalTrainingModel(args)
    print("Multimodal training model created successfully!")
