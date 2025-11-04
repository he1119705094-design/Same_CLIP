'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
''' 

import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from torchvision.datasets import ImageFolder
from PIL import ImageFile
from .cap_dset import CapDataset
from .processing import make_processing, add_processing_arguments
ImageFile.LOAD_TRUNCATED_IMAGES = True
np.random.seed(seed=17)

class ListToDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

class PathNameDataset(ImageFolder):
    def __init__(self, **keys):
        super().__init__(**keys)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return {"img": sample, "target": target, "path": path}

# opt是传入的所有参数，dataroot:D:/Model/AlignedForensics-master/datasets/valid
def get_dataset(opt, dataroot):
    dset_lst = []
    # NOTE: get the classes for the current directory
    # os.path.isdir的作用是：判断是否是目录
    if os.path.isdir(os.path.join(dataroot, "0_real")):#D:/Model/AlignedForensics-master/datasets/valid/0_real
        classes = ['.',] # classes是一个列表第一个元素是.
    else:
        classes = os.listdir(dataroot)# os.listdir输出所有文件和文件夹,返回值是一个列表
# transform = make_processing(opt) 这一行，实际就是构建图像预处理和增强的管道。
    # 它最终返回的是一个 torchvision.transforms.Compose 对象（或多视图增强封装），方便后续数据加载时对每张图片依次应用各种变换。
    transform = make_processing(opt) # transform接收来自数据预处理里面的所有操作。
    print('CLASSES:', classes)# 输出“CLASSES: ['airplane', 'bicycle', 'bird']”
    for cls in classes:
        root = dataroot + "/" + cls # D:/Model/AlignedForensics-master/datasets/valid/airplane
        if os.path.isdir(root + "/0_real"):
            dset = PathNameDataset(root=root, transform=transform)# 如果是多类结构（classes = ['cat', 'dog', 'car']），就会对每一类单独构造一个 PathNameDataset。
            print("#images %6d in %s" % (len(dset), root))# 输出images 长度 in 目录
            dset_lst.append(dset)                         # 返回一个列表有几类，列表中就有几个元素

    return torch.utils.data.ConcatDataset(dset_lst)# 将多个类别的数据集合并成一个统一的数据集对象，供 DataLoader 使用。
    #ConcatDataset([D1, D2, D3])会生成一个新的数据集 D，它像“拼接后的大数据集”，行为如下：
    #len(D) 等于所有子数据集长度之和。当你访问第 0 个样本到第 len(D1)-1 个样本时，它来自 D1。接着访问 D2 的样本。最后是 D3

def get_bal_sampler(dataset):
    targets = [0,1]
    #targets = []
    #for d in dataset.datasets:
    #    targets.extend(d.targets)

    ratio = np.bincount(targets)#  统计每个类别的样本数，比如 [900, 100]
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)# 计算每个类别的权重，样本越少权重越大
    if torch.all(w==w[0]):# 如果所有类别权重相同，直接用随机采样器RandomSampler
        print(f"RandomSampler: # {ratio}")# 随机采样表示每个类别数量相同
        sampler = RandomSampler(dataset, replacement = False)
    else:# 归一化权重，使和为1
        w = w / torch.sum(w)
        print(f"WeightedRandomSampler: # {ratio}, Weightes {w}")
        sample_weights = w[targets]# 给每个样本分配对应类别权重
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights)
        )
    return sampler


def add_dataloader_arguments(parser):
    # parser is an argparse.ArgumentParser
    #
    # This adds the arguments necessary for dataloader
    parser.add_argument(
        "--dataroot", type=str, help="Path to the dataset to use during training"
    )
    # The path containing the train and the validation data to train on
    parser.add_argument('--batched_syncing', action='store_true', help='synchronize the batches')
    parser.add_argument('--adm', action='store_true', help='account for ADM training')
    parser.add_argument("--data_cap", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_threads", default=8, type=int, help='# threads for loading data')
    parser.add_argument("--seed", default=8, type=int, help='# SEED')
    parser = add_processing_arguments(parser)
    return parser


def create_dataloader(opt, subdir='.', is_train=True):
    np.random.seed(seed=opt.seed)
    # 这是opt.dataroot的参数：D: / Model / AlignedForensics - master / datasets
    """
    这个时候opt里面的值：
    Namespace(name='first_test', arch='res50nodown', checkpoints_dir='./checkpoints/', pretrain=None, optim='adam', lr=0.0001, wei..., cmp_qual='30,100', resize_size=256, resize_ratio=0.75, norm_type='resnet', num_views=0, num_epoches=1000, earlystop_epoch=10)
    """
    dataroot = os.path.join(opt.dataroot, subdir) # D:/Model/AlignedForensics-master/datasets/valid
    # 传入的opt.data_cap:1000,opt.batched_syncing:None，is_train：false,因为我先看的valid
    # A is not B：判断对象 A 和 B 是否不是同一个对象（注意：是身份比较，不是值比较）
    if (opt.data_cap is not None or opt.batched_syncing) and is_train:
        transform = make_processing(opt)
        dataset = CapDataset(root_dir=dataroot, data_cap=opt.data_cap,transform=transform,batched_syncing=opt.batched_syncing,use_inversions=opt.use_inversions,seed=opt.seed)
        dataset = torch.utils.data.ConcatDataset([dataset])
    else:# 是用于验证集使用的，因为is_train:False
        dataset = get_dataset(opt, dataroot) #经过get_dataset()函数之后，验证集中的几个数据集会拼在一起，形成一个大的数据集，dataset接收的就是这个大的数据集
    data_loader = torch.utils.data.DataLoader(
        dataset,                            #这是前面构建好的数据集对象，多个数据集合并（ConcatDataset）。
        batch_size=opt.batch_size,          #指定每个批次加载多少张图片，比如你传入的是8，那么每次从数据集中取8张图片形成一个 batch。
        sampler=get_bal_sampler(dataset) if is_train else None,#这里根据是否是训练集 (is_train) 决定是否使用采样器。
                                            #解释采样器：采样器（Sampler） 就是控制从数据集中按照什么规则、顺序、频率去采样数据的组件。
                                            #如果是训练集，会调用get_bal_sampler(dataset)生成一个平衡采样器（Balanced Sampler），用于按一定权重或均衡方式采样，避免类别不平衡带来的训练偏差。
                                            #如果是验证集，则不使用采样器，默认顺序采样。
        #num_workers=int(opt.num_threads),  #说明代码支持根据配置动态设置线程数，不过这里固定用0。
        num_workers=16,# 指明加载数据时使用多少个子进程并行加载。这里设为0，表示数据加载在主进程同步进行，没有使用多线程或多进程。
        pin_memory=True,  # ✅ 提高主机到GPU的传输速度
        prefetch_factor=2,  # ✅ Python 3.8+ 支持
        persistent_workers=True  # ✅ 只在PyTorch 1.8+可用
    )
    return data_loader
