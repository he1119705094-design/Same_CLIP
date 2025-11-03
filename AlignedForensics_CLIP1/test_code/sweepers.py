import torch
import os
import pandas
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.processing import make_normalize
import matplotlib.pyplot as plt
import tqdm
import math
import glob
import sys
import yaml
import io
import cv2
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from networks import create_architecture, load_weights
from dataset import UnlabeledImageDataset
import torchvision.transforms as transforms
import statistics

def create_scale_list(start_res, end_res, step, isQual=False):
    num_steps = (end_res - start_res) // step + 1
    if isQual:
        scale_list = [start_res + i*step for i in range(num_steps)]
    else:
        scale_list = [(start_res + i*step)/start_res for i in range(num_steps)]
    return scale_list

def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])

    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']



def get_models(weights_dir, models_list, device, use_proj=False):
    norm_type = 'resnet'
    models_dict = dict()
    print("Models:")
    for model_name in models_list:
        print(model_name, flush=True)
        _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)
        print("SIZE:", patch_size)
        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()
        models_dict[model_name] = model
        print(flush=True)
    return models_dict

def run_models_adv(models_dict, loader, device, raw=False):
    probs = {}
    for images in loader:
        for name in models_dict.keys():
            if name not in probs:
                probs[name] = []

            scores = models_dict[name](images.to(device)).cpu()
            probs[name].extend(torch.sigmoid(scores).tolist())
    
    for key in probs:
        p = np.array(probs[key]).flatten()
        if raw:
            probs[key] = p.tolist()
        else:
            # Compute required spread metrics
            mean = statistics.mean(p)
            std_dev = statistics.stdev(p)
            p25 = np.percentile(p, 25)
            p75 = np.percentile(p, 75)
            iqr = p75 - p25
            p5, p95 = np.percentile(p, 5), np.percentile(p, 95)

            # Store metrics in a dictionary
            probs[key] = {
                "mean": mean,
                "sd": std_dev,
                "iqr": iqr,
                "25th_percentile": p25,
                "75th_percentile": p75,
                "5th_percentile": p5,
                "95th_percentile": p95
            }
    return probs

def webp_compress(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=quality)
    return Image.open(buffer)

def load_datasets_with_compression(models_list, real_folder, fake_folder, start_quality, end_quality, step, args):
    models_dict = get_models(weights_dir=args['weights_dir'], models_list=models_list, device=args['device'])
    real_probs = {}
    fake_probs = {}

    for key in models_dict.keys():
        real_probs[key] = {'mean': [], 'sd': [], "iqr": [], "25th_percentile": [],
                           "5th_percentile": [], "75th_percentile": [], "95th_percentile": []}
        fake_probs[key] = {'mean': [], 'sd': [], "iqr": [], "25th_percentile": [],
                           "5th_percentile": [], "75th_percentile": [], "95th_percentile": []}
    
    for quality in range(start_quality, end_quality + 1, step):
        print(f"Loading datasets with WebP compression quality {quality}")

        transform = transforms.Compose([
            transforms.Lambda(lambda img: webp_compress(img, quality)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        real_dataset = UnlabeledImageDataset(real_folder, transform=transform)
        fake_dataset = UnlabeledImageDataset(fake_folder, transform=transform)

        # Adjust batch size according to the quality if necessary
        bs = 20
        real_loader = DataLoader(real_dataset, batch_size=bs, shuffle=True)
        fake_loader = DataLoader(fake_dataset, batch_size=bs, shuffle=True)

        with torch.no_grad():
            real_stats = run_models_adv(models_dict, real_loader, device=args['device'])
            fake_stats = run_models_adv(models_dict, fake_loader, device=args['device'])
            
        for key in models_dict.keys():
            for key_in in real_stats[key].keys():
                real_probs[key][key_in].append(real_stats[key][key_in])
                fake_probs[key][key_in].append(fake_stats[key][key_in])

    return real_probs, fake_probs



def create_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC, antialias=True),  # Resize images to the specified resolution
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#RESNET NORM
    ])


def load_datasets_with_resolutions(models_list,real_folder, fake_folder, start_res, end_res, step,args):
    models_dict = get_models(weights_dir=args['weights_dir'], models_list=models_list, device=args['device'], use_proj=args['use_proj'])
    real_probs = {}
    fake_probs = {}
    for key in models_dict.keys():
        real_probs[key] = {'mean': [], 'sd': [], "iqr": [], "25th_percentile": [],
                           "5th_percentile": [], "75th_percentile": [], "95th_percentile": []}
        fake_probs[key] = {'mean': [], 'sd': [], "iqr": [], "25th_percentile": [],
                           "5th_percentile": [], "75th_percentile": [], "95th_percentile": []}
        
    for res in range(start_res, end_res + 1, step):
        print(f"Loading datasets with resolution {res}x{res}")
        transform = create_transform(res)

        real_dataset = UnlabeledImageDataset(real_folder, transform=transform)
        fake_dataset = UnlabeledImageDataset(fake_folder, transform=transform)
        #Adjust the batch size to fit on the machine
        if res <300:
            bs=80
        elif res<500:
            bs=30
        else:
            bs=5
        real_loader = DataLoader(real_dataset, batch_size=bs, shuffle=True)
        fake_loader = DataLoader(fake_dataset, batch_size=bs, shuffle=True)
        
        with torch.no_grad():
            real_stats = run_models_adv(models_dict, real_loader, device=args['device'])
            fake_stats = run_models_adv(models_dict, fake_loader, device=args['device'])

        for key in models_dict.keys():
            for key_in in real_stats[key].keys():
                real_probs[key][key_in].append(real_stats[key][key_in])
                fake_probs[key][key_in].append(fake_stats[key][key_in])
        
    return real_probs, fake_probs

