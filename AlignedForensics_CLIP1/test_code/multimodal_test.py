"""
多模态测试脚本
支持CLIP增强的假图像检测模型测试
"""

import os
import torch
import pandas as pd
import numpy as np
import tqdm
import yaml
from PIL import Image
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode

# 添加训练代码路径
import sys
sys.path.append('../training_code')
# 获取项目根目录 (AlignedForensics_CLIP)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from training_code.utils.multimodal_training import MultimodalTrainingModel
from training_code.utils.captioning import ImageCaptioner
from utils.processing import make_normalize
from networks import create_architecture, load_weights

class MultimodalTester:
    """多模态测试器"""
    
    def __init__(self, weights_dir, device="cuda:0"):
        self.device = device
        self.weights_dir = weights_dir
        self.models = {}
        self.captioner = None
        
    def load_model(self, model_name, enable_captioning=False):
        """加载多模态模型"""
        config_path = os.path.join(self.weights_dir, model_name, 'config.yaml')
        
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return False
        
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 创建模型参数
        class Args:
            def __init__(self, config):
                for key, value in config.items():
                    setattr(self, key, value)
                # 设置默认值
                self.enable_multimodal = True
                self.fusion_method = "attention"
                self.hidden_dim = 512
                self.freeze_clip = True
                self.enable_captioning = enable_captioning
                self.enable_feature_optimization = False
                self.enable_visualization = False
        
        args = Args(config)
        
        # 创建多模态模型
        model = MultimodalTrainingModel(args, subdir=model_name)
        
        # 加载权重
        model_path = os.path.join(self.weights_dir, model_name, config['weights_file'])
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model' in checkpoint:
                model.multimodal_detector.load_state_dict(checkpoint['model'])
            else:
                model.multimodal_detector.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        self.models[model_name] = model
        
        # 初始化图像描述生成器
        if enable_captioning:
            self.captioner = ImageCaptioner(device=self.device)
        
        return True
    
    def test_single_image(self, image_path, model_name, generate_caption=False):
        """测试单张图像"""
        if model_name not in self.models:
            print(f"Model {model_name} not loaded")
            return None
        
        model = self.models[model_name]
        
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        transform = Compose([
            Resize(256, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            make_normalize('resnet')
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # 生成描述（如果需要）
        caption = None
        if generate_caption and self.captioner:
            try:
                caption = self.captioner.generate_caption(image)
            except Exception as e:
                print(f"Caption generation failed: {e}")
                caption = ""
        
        # 预测
        with torch.no_grad():
            if hasattr(model, 'multimodal_detector'):
                output = model.multimodal_detector(image_tensor, [caption] if caption else None)
            else:
                output = model(image_tensor)
            
            # 获取预测分数
            if len(output.shape) > 1:
                score = output[0, -1].item()
            else:
                score = output[0].item()
        
        return {
            'image_path': image_path,
            'score': score,
            'prediction': 'fake' if score < 0 else 'real',
            'caption': caption
        }
    
    def test_batch(self, image_paths, model_name, batch_size=32, generate_captions=False):
        """批量测试图像"""
        if model_name not in self.models:
            print(f"Model {model_name} not loaded")
            return []
        
        model = self.models[model_name]
        results = []
        
        # 预处理图像
        transform = Compose([
            Resize(256, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            make_normalize('resnet')
        ])
        
        for i in tqdm.tqdm(range(0, len(image_paths), batch_size), desc="Testing images"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_captions = []
            
            # 加载批次图像
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = transform(image)
                    batch_images.append(image_tensor)
                    
                    # 生成描述
                    if generate_captions and self.captioner:
                        try:
                            caption = self.captioner.generate_caption(image)
                            batch_captions.append(caption)
                        except:
                            batch_captions.append("")
                    else:
                        batch_captions.append(None)
                        
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # 转换为tensor
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # 预测
            with torch.no_grad():
                if hasattr(model, 'multimodal_detector'):
                    output = model.multimodal_detector(batch_tensor, batch_captions)
                else:
                    output = model(batch_tensor)
                
                # 处理输出
                if len(output.shape) > 1:
                    scores = output[:, -1].cpu().numpy()
                else:
                    scores = output.cpu().numpy()
            
            # 保存结果
            for j, (path, score) in enumerate(zip(batch_paths, scores)):
                results.append({
                    'image_path': path,
                    'score': float(score),
                    'prediction': 'fake' if score < 0 else 'real',
                    'caption': batch_captions[j] if batch_captions[j] else ""
                })
        
        return results
    
    def compare_models(self, image_paths, model_names, generate_captions=False):
        """比较多个模型的性能"""
        all_results = {}
        
        for model_name in model_names:
            print(f"Testing with model: {model_name}")
            results = self.test_batch(image_paths, model_name, generate_captions=generate_captions)
            all_results[model_name] = results
        
        # 创建比较DataFrame
        comparison_data = []
        for i, path in enumerate(image_paths):
            row = {'image_path': path}
            for model_name in model_names:
                if i < len(all_results[model_name]):
                    result = all_results[model_name][i]
                    row[f'{model_name}_score'] = result['score']
                    row[f'{model_name}_prediction'] = result['prediction']
                    if generate_captions:
                        row[f'{model_name}_caption'] = result['caption']
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def create_interpretability_report(self, image_paths, model_name, output_dir="interpretability"):
        """创建可解释性报告"""
        if model_name not in self.models:
            print(f"Model {model_name} not loaded")
            return
        
        model = self.models[model_name]
        
        if not hasattr(model, 'create_interpretability_visualization'):
            print("Model does not support interpretability visualization")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建数据加载器
        from torch.utils.data import DataLoader, TensorDataset
        
        # 预处理图像
        transform = Compose([
            Resize(256, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            make_normalize('resnet')
        ])
        
        images = []
        for path in image_paths[:4]:  # 只可视化前4张图像
            try:
                image = Image.open(path).convert('RGB')
                image_tensor = transform(image)
                images.append(image_tensor)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        if not images:
            print("No valid images found for visualization")
            return
        
        images_tensor = torch.stack(images).to(self.device)
        
        # 生成描述
        captions = None
        if self.captioner:
            captions = []
            for path in image_paths[:4]:
                try:
                    image = Image.open(path).convert('RGB')
                    caption = self.captioner.generate_caption(image)
                    captions.append(caption)
                except:
                    captions.append("")
        
        # 创建可视化
        model.create_interpretability_visualization(
            [(images_tensor, torch.zeros(len(images_tensor)))],  # 模拟数据加载器
            output_dir
        )

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal Fake Image Detection Testing")
    parser.add_argument("--weights_dir", type=str, required=True,
                       help="Directory containing model weights")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Name of the model to test")
    parser.add_argument("--image_paths", type=str, nargs='+',
                       help="Paths to images to test")
    parser.add_argument("--csv_file", type=str,
                       help="CSV file containing image paths")
    parser.add_argument("--output_csv", type=str, default="multimodal_results.csv",
                       help="Output CSV file for results")
    parser.add_argument("--enable_captioning", action="store_true",
                       help="Enable image captioning")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use for testing")
    parser.add_argument("--create_visualization", action="store_true",
                       help="Create interpretability visualizations")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = MultimodalTester(args.weights_dir, args.device)
    
    # 加载模型
    if not tester.load_model(args.model_name, args.enable_captioning):
        print("Failed to load model")
        return
    
    # 获取图像路径
    if args.csv_file:
        df = pd.read_csv(args.csv_file)
        image_paths = df['filename'].tolist()
    else:
        image_paths = args.image_paths
    
    if not image_paths:
        print("No images to test")
        return
    
    # 测试图像
    print(f"Testing {len(image_paths)} images...")
    results = tester.test_batch(
        image_paths, 
        args.model_name, 
        batch_size=args.batch_size,
        generate_captions=args.enable_captioning
    )
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    
    # 创建可视化
    if args.create_visualization:
        print("Creating interpretability visualizations...")
        tester.create_interpretability_report(
            image_paths[:4],  # 只可视化前4张图像
            args.model_name,
            output_dir="interpretability"
        )
    
    # 打印统计信息
    real_count = sum(1 for r in results if r['prediction'] == 'real')
    fake_count = sum(1 for r in results if r['prediction'] == 'fake')
    avg_score = np.mean([r['score'] for r in results])
    
    print(f"\nTest Results:")
    print(f"Total images: {len(results)}")
    print(f"Predicted real: {real_count}")
    print(f"Predicted fake: {fake_count}")
    print(f"Average score: {avg_score:.4f}")

if __name__ == "__main__":
    main()
