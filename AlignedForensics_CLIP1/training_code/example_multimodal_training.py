"""
多模态训练示例脚本
演示如何使用CLIP增强的假图像检测系统
"""

import os
import torch
import argparse
from utils.multimodal_training import MultimodalTrainingModel, add_multimodal_training_arguments
from utils.dataset import create_dataloader

def main():
    parser = argparse.ArgumentParser(description="Multimodal Fake Image Detection Training Example")
    parser = add_multimodal_training_arguments(parser)
    
    # 基础参数
    parser.add_argument("--name", type=str, default="multimodal_experiment")
    parser.add_argument("--dataroot", type=str, required=True, help="Path to dataset")
    parser.add_argument("--arch", type=str, default="res50nodown")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epoches", type=int, default=10)
    parser.add_argument("--earlystop_epoch", type=int, default=5)
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    
    args = parser.parse_args()
    
    print("=== 多模态假图像检测训练示例 ===")
    print(f"实验名称: {args.name}")
    print(f"数据集路径: {args.dataroot}")
    print(f"多模态模式: {args.enable_multimodal}")
    print(f"图像描述: {args.enable_captioning}")
    print(f"特征优化: {args.enable_feature_optimization}")
    print(f"融合方法: {args.fusion_method}")
    print(f"可视化: {args.enable_visualization}")
    print("=" * 50)
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader = create_dataloader(args, subdir="train", is_train=True)
    valid_loader = create_dataloader(args, subdir="valid", is_train=False)
    
    print(f"训练批次数量: {len(train_loader)}")
    print(f"验证批次数量: {len(valid_loader)}")
    
    # 创建多模态训练模型
    print("创建多模态训练模型...")
    model = MultimodalTrainingModel(args, subdir=args.name)
    
    # 显示模型信息
    if hasattr(model, 'multimodal_detector'):
        print("多模态检测器已创建")
        print(f"融合方法: {model.fusion_method}")
        print(f"隐藏维度: {model.hidden_dim}")
    
    if hasattr(model, 'captioner'):
        print("图像描述生成器已启用")
    
    if hasattr(model, 'feature_optimizer'):
        print("特征优化器已启用")
    
    if hasattr(model, 'visualizer'):
        print("可视化工具已启用")
    
    # 模拟训练过程
    print("\n开始模拟训练...")
    model.train()
    
    for epoch in range(min(3, args.num_epoches)):  # 只运行3个epoch作为示例
        print(f"\nEpoch {epoch + 1}/{args.num_epoches}")
        
        # 训练
        train_losses = []
        for i, data in enumerate(train_loader):
            if i >= 5:  # 只处理前5个批次作为示例
                break
            
            loss = model.train_on_batch(data)
            train_losses.append(loss.item())
            print(f"  Batch {i+1}/5, Loss: {loss.item():.4f}")
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"  平均训练损失: {avg_train_loss:.4f}")
        
        # 验证
        print("  验证中...")
        model.eval()
        y_true, y_pred, y_path = model.predict(valid_loader)
        
        # 计算准确率
        from sklearn.metrics import balanced_accuracy_score
        acc = balanced_accuracy_score(y_true, y_pred > 0.0)
        print(f"  验证准确率: {acc:.4f}")
        
        # 多模态组件评估
        if hasattr(model, 'evaluate_multimodal_components'):
            component_metrics = model.evaluate_multimodal_components(valid_loader)
            for metric_name, metric_value in component_metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")
    
    # 创建可解释性可视化
    if hasattr(model, 'create_interpretability_visualization'):
        print("\n创建可解释性可视化...")
        model.create_interpretability_visualization(
            valid_loader, 
            output_dir=os.path.join(model.save_dir, "interpretability")
        )
        print(f"可视化结果保存在: {os.path.join(model.save_dir, 'interpretability')}")
    
    # 保存模型
    print("\n保存模型...")
    model.save_networks('final')
    print(f"模型保存在: {model.save_dir}")
    
    print("\n=== 训练完成 ===")
    print("您可以使用以下命令进行测试:")
    print(f"python ../test_code/multimodal_test.py --weights_dir {model.save_dir} --model_name {args.name}")

if __name__ == "__main__":
    main()
