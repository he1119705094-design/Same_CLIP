
import torch
import os
import pandas
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.processing import make_normalize
import matplotlib.pyplot as plt
import tqdm
import glob
import sys
import yaml
import json
import io
from PIL import Image
import seaborn as sns
from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from networks import create_architecture, load_weights
from dataset import UnlabeledImageDataset
import torchvision.transforms as transforms
import statistics
from sweepers import *

def plot_means(models_dict, x_values, x_label, y_label, title, filename):
    plt.figure(figsize=(10, 6))

    for model_name, stats in models_dict.items():
        # Plot mean line
        plt.plot(x_values, stats['mean'], label=model_name)
        # Plot mean ± standard deviation as a shaded area
        if 'sd' in stats:
            mean = np.array(stats['mean'])
            sd = np.array(stats['sd'])
            plt.fill_between(
                x_values, mean - sd, mean + sd,
                color=plt.gca().lines[-1].get_color(), alpha=0.2,
                label=f'{model_name} ±1 SD'
            )

    # Set plot properties
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save and close plot
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, help="The real folder")
    parser.add_argument("--fake", type=str, help="The fake folder")
    parser.add_argument("--out_path", type=str, help="Path to save the plot", default='plots')
    parser.add_argument("--setting", type=str, help="Enter either res or qual, based on whether to sweep resolution or quality", default='res')
    parser.add_argument("--weights_dir", type=str, help="The directory to the networks weights", default="./weights/")
    parser.add_argument("--models", type=str, help="List of models to test", default='ours,ours-sync')
    parser.add_argument("--device", type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--start", type=float, help="Starting setting", default=256)
    parser.add_argument("--end", type=float, help="Final setting", default=1024)
    parser.add_argument("--stride", type=int, help="Stride", default=2)
    parser.add_argument("--base_res", type=int, help="ogres", default=512)
    parser.add_argument('--use_proj', action='store_true', help='Use a projection layer, before contrastive training')
    args = vars(parser.parse_args())
    
    if args['models'] is None:
        args['models'] = os.listdir(args['weights_dir'])
    else:
        args['models'] = args['models'].split(',')
    if args['setting'] == 'res':
        real_probs, fake_probs = load_datasets_with_resolutions(models_list=args['models'],real_folder=args['real'], 
                                                                fake_folder=args['fake'], start_res=int(args['start']), 
                                                                end_res=int(args['end']), step=args['stride'], args=args)
        x = create_scale_list(start_res=int(args['start']), end_res=int(args['end']), step=args['stride'])
        sweeper = 'Scaling Factor'
    elif args['setting'] == 'qual':
        real_probs, fake_probs = load_datasets_with_compression(models_list=args['models'], real_folder=args['real'], 
                                                                fake_folder=args['fake'], start_quality=int(args['start']),
                                                                end_quality=int(args['end']), step=args['stride'], args=args)
        x = list(range(int(args['start']), int(args['end'])+1, args['stride']))
        sweeper = 'Webp compression quality'

    x = [t * (args['start']/args['base_res']) for t in x]
    if args['setting'] == 'res':
        y_label = 'Probability of Image Being Fake'
        x_label = 'Scaling Factor'
    elif args['setting'] == 'qual':
        x = list(range(int(args['start']), int(args['end'])+1, int(args['stride'])))        
        y_label = 'Probability of Image Being Fake'
        x_label = 'Scaling Factor'
        plot_means(models_dict=real_probs, x_values=x, x_label='Compression Quality',
                   y_label='Probability of Image Being Fake', title='Real Images', 
                   filename=os.path.join(args["out_path"],'real_means.png'))
        plot_means(models_dict=fake_probs, x_values=x, x_label='Compression Quality', y_label='Probability of Image Being Fake', title='Fake Images', filename=os.path.join(args["out_path"],'fake_means.png'))
    
    plot_means(models_dict=real_probs, x_values=x, x_label=x_label,
                   y_label=y_label, title='Real Images', 
                   filename=os.path.join(args["out_path"],'real_means.png'))
    plot_means(models_dict=fake_probs, x_values=x, x_label=x_label,
                   y_label=y_label, title='Fake Images',
                   filename=os.path.join(args["out_path"],'fake_means.png'))
