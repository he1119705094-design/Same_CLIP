"""
多模态模型评估脚本
按照原始eval.py的格式，支持多模态模型评估
"""

import argparse
from copy import deepcopy
import os
import math
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import sys
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def statistics(data):
    """计算统计信息"""
    if not data:
        print("List is empty")
        return

    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    maximum = max(data)
    minimum = min(data)
    
    print(f"Mean: {mean:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"Max: {maximum:.4f}")
    print(f"Min: {minimum:.4f}")

def read_csv_files(file1, file2):
    """读取CSV文件"""
    df1 = pd.read_csv(file1, nrows=200000)
    df2 = pd.read_csv(file2, nrows=200000)
    return df1, df2

def plot_roc_curve(y_true, y_scores, model_name, output_dir="eval_results"):
    """绘制ROC曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return auc

def main():
    """主评估函数"""
    parser = argparse.ArgumentParser(description='Evaluate multimodal models.')
    parser.add_argument('--real', type=str, help='file containing scores of the real dataset')
    parser.add_argument('--fake', type=str, help='file containing scores of the fake dataset')
    parser.add_argument('--ix', type=int, default=1, help='number of models to evaluate')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='output directory for results')
    parser.add_argument('--plot_roc', action='store_true', help='plot ROC curves')
    
    args = parser.parse_args()
    thresholds = {}

    df_real, df_fake = read_csv_files(args.real, args.fake)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 评估每个模型
    for model in df_real.columns[-args.ix:]:
        thresholds[model] = 0.50
        print(f'\n=== 评估模型: {model} ===')
        print('THRESHOLD:', thresholds[model])
        print('-' * 80)
        
        # 获取分数并应用sigmoid
        real_scores = F.sigmoid(torch.tensor(df_real[model].tolist()))
        real_scores = [x.item() for x in real_scores if not math.isnan(x.item())]
        
        fake_scores = F.sigmoid(torch.tensor(df_fake[model].tolist()))
        fake_scores = [x.item() for x in fake_scores if not math.isnan(x.item())]
        
        all_scores = real_scores + fake_scores
        
        # 打印统计信息
        print('真实图像统计:')
        statistics(data=real_scores)
        print('\n假图像统计:')
        statistics(data=fake_scores)
        print('-' * 80)
        
        # 计算预测结果
        preds = (np.array(all_scores) >= thresholds[model]).astype(int)
        real_true = [0] * len(real_scores) 
        fake_true = [1] * len(fake_scores)
        y_true = real_true + fake_true
        
        # 计算各种指标
        real_acc = accuracy_score(real_true, (np.array(real_scores) >= thresholds[model]).astype(int))
        fake_acc = accuracy_score(fake_true, (np.array(fake_scores) >= thresholds[model]).astype(int))
        overall_acc = accuracy_score(y_true, preds)
        
        # 计算精确率、召回率、F1分数
        precision = precision_score(y_true, preds)
        recall = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        
        # 计算AP和AUC
        ap = average_precision_score(y_true, all_scores)
        auc = roc_auc_score(y_true, all_scores)
        
        print(f'{model} 真实图像准确率: {real_acc:.4f}')
        print(f'{model} 假图像准确率: {fake_acc:.4f}')
        print(f'{model} 总体准确率: {overall_acc:.4f}')
        print(f'{model} 精确率: {precision:.4f}')
        print(f'{model} 召回率: {recall:.4f}')
        print(f'{model} F1分数: {f1:.4f}')
        print(f'{model} AP: {ap:.4f}')
        print(f'{model} AUC: {auc:.4f}')
        print('-' * 80)
        
        # 绘制ROC曲线
        if args.plot_roc:
            plot_roc_curve(y_true, all_scores, model, args.output_dir)
        
        # 保存详细结果
        results = {
            'model': model,
            'real_accuracy': real_acc,
            'fake_accuracy': fake_acc,
            'overall_accuracy': overall_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'ap': ap,
            'auc': auc,
            'threshold': thresholds[model]
        }
        
        results_df = pd.DataFrame([results])
        results_df.to_csv(os.path.join(args.output_dir, f'{model}_results.csv'), index=False)
    
    print(f"\n评估完成！结果保存在: {args.output_dir}")

if __name__ == '__main__':
    main()

