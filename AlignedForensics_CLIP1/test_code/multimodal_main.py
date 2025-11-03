"""
多模态模型测试主脚本
按照原始main.py的格式，支持多模态模型测试
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import pandas
import numpy as np
import tqdm
import glob
import sys
import yaml
from PIL import Image
from PIL.ImageFile import ImageFile

from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from networks import create_architecture, load_weights
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 添加训练代码路径以导入多模态组件
sys.path.append('../training_code')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from training_code.utils.multimodal_training import MultimodalTrainingModel
from training_code.utils.dataset import create_dataloader

def get_multimodal_config(model_name, weights_dir='./weights'):
    """获取多模态模型配置"""
    config_path = os.path.join(weights_dir, model_name, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path) as fid:
            data = yaml.load(fid, Loader=yaml.FullLoader)
        model_path = os.path.join(weights_dir, model_name, data['weights_file'])
        return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']
    else:
        # 默认配置
        return 'multimodal_ours', os.path.join(weights_dir, model_name, 'model_epoch_best.pth'), 'res50nodown', 'resnet', 96

def run_multimodal_tests(input_csv, weights_dir, models_list, device, batch_size=1, fusion_method='concat'):
    """运行多模态模型测试"""
    table = pandas.read_csv(input_csv)[['filename',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    
    models_dict = dict()
    transform_dict = dict()
    print("Multimodal Models:")
    
    for model_name in models_list:
        print(f"Loading {model_name}...", flush=True)
        _, model_path, arch, norm_type, patch_size = get_multimodal_config(model_name, weights_dir=weights_dir)

        # 创建多模态模型
        class TestArgs:
            def __init__(self):
                self.arch = arch
                self.norm_type = norm_type
                self.cropSize = patch_size if patch_size else 96
                self.resize_size = 256
                self.resize_ratio = 0.75
                self.blur_sig = "0.0,3.0"
                self.cmp_method = "cv2,pil"
                self.cmp_qual = "30,100"
                self.resize_prob = 0.0
                self.jitter_prob = 0.0
                self.colordist_prob = 0.0
                self.cutout_prob = 0.0
                self.noise_prob = 0.0
                self.blur_prob = 0.0
                self.cmp_prob = 0.0
                self.rot90_prob = 0.0
                self.batched_syncing = False
                self.use_inversions = False
                self.data_cap = None
                self.enable_multimodal = True
                self.fusion_method = fusion_method
                self.enable_visualization = False
                # ✅ 以下是新增的字段
                # 新增：必须提供checkpoints_dir，否则training.py会报错
                self.checkpoints_dir = weights_dir
                self.no_cuda = False  # 表示允许使用GPU
                self.start_fresh = False  # 不重新初始化模型，直接加载预训练
                self.use_leaky = False  # 不启用 leaky ReLU
                self.ckpt = None  # 无需单独 checkpoint
                self.use_proj = False  # 一般用于多模态对齐投影层，这里禁用
                self.proj_ratio = 1.0  # 投影比例
                self.final_dropout = 0.0  # 输出层 dropout

        args = TestArgs()
        
        # 创建多模态训练模型
        model = MultimodalTrainingModel(args, subdir=f"multimodal_test_{model_name}")
        
        # 加载权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model' in checkpoint:
                model.multimodal_detector.load_state_dict(checkpoint['model'],strict=False)
            else:
                model.multimodal_detector.load_state_dict(checkpoint)
            print(f"Model weights loaded: {model_path}")
        else:
            print(f"Warning: Model file not found {model_path}")
            continue

        model = model.to(device).eval()

        # 设置图像变换
        transform = list()
        if patch_size is None:
            print('input none', flush=True)
            transform_key = 'none_%s' % norm_type
        elif patch_size == 'Clip224':
            print('input resize:', 'Clip224', flush=True)
            transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
            transform.append(CenterCrop((224, 224)))
            transform_key = 'Clip224_%s' % norm_type
        elif isinstance(patch_size, tuple) or isinstance(patch_size, list):
            print('input resize:', patch_size, flush=True)
            transform.append(Resize(*patch_size))
            transform.append(CenterCrop(patch_size[0]))
            transform_key = 'res%d_%s' % (patch_size[0], norm_type)
        elif patch_size > 0:
            print('input crop:', patch_size, flush=True)
            transform.append(CenterCrop(patch_size))
            transform_key = 'crop%d_%s' % (patch_size, norm_type)
        
        transform.append(make_normalize(norm_type))
        transform = Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)
        print(flush=True)

    ### 测试
    with torch.no_grad():
        do_models = list(models_dict.keys())
        do_transforms = set([models_dict[_][0] for _ in do_models])
        print("Models to test:", do_models)
        print("Transforms:", do_transforms)
        print(flush=True)
        
        print("Running Multimodal Tests")
        batch_img = {k: list() for k in transform_dict}
        batch_id = list()
        last_index = table.index[-1]
        
        for index in tqdm.tqdm(table.index, total=len(table)):
            raw_path = str(table.loc[index, 'filename'])
            filename = raw_path if os.path.isabs(raw_path) else os.path.join(rootdataset, raw_path)
            img_name = os.path.splitext(os.path.basename(filename))[0]
            
            try:
                img = Image.open(filename).convert('RGB')
                for k in transform_dict:
                    batch_img[k].append(transform_dict[k](img))
                batch_id.append(index)
            except Exception as e:
                print(f"[跳过] 无法读取图像: {filename}，错误：{e}")
                continue

            if (len(batch_id) >= batch_size) or (index == last_index):
                for k in do_transforms:
                    batch_img[k] = torch.stack(batch_img[k], 0)

                for model_name in do_models:
                    model = models_dict[model_name][1]
                    model.current_img_name = img_name
                    
                    # 多模态推理
                    images = batch_img[models_dict[model_name][0]].clone().to(device)
                    captions = None  # 暂时不使用文本描述
                    
                    # 获取多模态输出
                    output = model.multimodal_detector(images, captions)
                    
                    if isinstance(output, dict):
                        out_tens = output['logits'].cpu().numpy()
                    else:
                        out_tens = output.cpu().numpy()
                    
                    if out_tens.shape[1] == 1:
                        out_tens = out_tens[:, 0]
                    elif out_tens.shape[1] == 2:
                        out_tens = out_tens[:, 1] - out_tens[:, 0]
                    else:
                        assert False
                    
                    if len(out_tens.shape) > 1:
                        logit1 = np.mean(out_tens, (1, 2))
                    else:
                        logit1 = out_tens

                    for ii, logit in zip(batch_id, logit1):
                        table.loc[ii, model_name] = logit

                batch_img = {k: list() for k in transform_dict}
                batch_id = list()

            torch.cuda.empty_cache()
            assert len(batch_id) == 0
        
    # 若未加载任何模型，给出提示，避免只输出filename列
    if len(models_dict) == 0:
        print("Warning: No models were loaded. Please check weights_dir/models names.")
    return table

if __name__ == "__main__":
    # 使用argparse解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal model testing")
    parser.add_argument("--in_csv", '-i', type=str, help="The path of the input csv file with the list of images")
    parser.add_argument("--out_csv", '-o', type=str, help="The path of the output csv file", default="./multimodal_results.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="./weights")
    parser.add_argument("--models", '-m', type=str, help="List of models to test", default='multimodal_ours')
    parser.add_argument("--fusion", '-f', type=str, help="Fusion method", default='concat')
    parser.add_argument("--device", '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    
    args = parser.parse_args()
    
    # 如果models是字符串，转为列表
    if isinstance(args.models, str):
        args.models = args.models.split(',')
    
    table = run_multimodal_tests(args.in_csv, args.weights_dir, args.models, args.device, args.batch_size, args.fusion)
    
    output_csv = args.out_csv
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    table.to_csv(output_csv, index=False)  # 保存结果为CSV文件
