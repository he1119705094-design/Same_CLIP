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
import sys

import torch
import torch.nn as nn
import numpy as np
import tqdm
from .losses import SupConLoss
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from training_code.networks import create_architecture, count_parameters # .. 表示回到上一层目录，即 training_code/
#from ..networks import create_architecture, count_parameters
import matplotlib.pyplot as plt
import pprint
import random
import torch.nn.init as init





def add_training_arguments(parser):
    # parser is an argparse.ArgumentParser
    #
    # This adds the arguments necessary for the training

    parser.add_argument(
        "--arch", type=str, default="res50nodown",
        help="architecture name"
    )
    parser.add_argument(
        "--checkpoints_dir",
        default="./checkpoints/",
        type=str,
        help="Path to the dataset to use during training",
    )
    parser.add_argument("--pretrain", type=str, default=None, help="pretrained weights")
    parser.add_argument(
        "--optim", type=str, default="adam", help="optim to use [sgd, adam]"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight decay"
    )
    parser.add_argument("--ckpt", type=str, default=None, help="path to load some custom weights")
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="momentum term of adam"
    )
    parser.add_argument(
    "--eps_adv", type=float, nargs='+', default=None, help="epsilon values for FGSM-based adversarial training"
    )
    parser.add_argument(
                "--lambda_bce", type=float, default=None, help="Weight of bce loss when performing contrastive training"
                    )
    parser.add_argument(
        "--proj_ratio", type=int, default=None, help="Factor to scale down the 2048 dimensional space"
    )
    parser.add_argument('--start_fresh', action='store_true', help='Setting this true makes the training start from random weights, not pretrained')
    parser.add_argument('--use_leaky', action='store_true', help='Use leaky ReLU to potentially avoid dying relu')
    parser.add_argument('--flex_rz', action='store_true', help='Use random resized crop to all kinds of resolution from 128-1024')
    parser.add_argument('--only_coco', action='store_true', help='Use only the coco dataset')
    parser.add_argument('--use_contrastive', action='store_true', help='Use contrastive learning on the penultimate layer')
    parser.add_argument('--use_proj', action='store_true', help='Use a projection layer, before contrastive training')
    parser.add_argument('--use_inversions', action='store_true', help='Use Inversions to train')
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="run on CPU",
    )

    parser.add_argument(
        "--continue_epoch",
        type=str,
        default=None,
        help="Whether the network is going to be trained",
    )

    # ICML paper settings
    parser.add_argument('--fix_backbone', action='store_true', help='Perform linear probing while freezing the rest of the model')
    parser.add_argument('--stay_positive', type=str, default=None, help='Whether to rely on sigmoiding the weights or clamping them')
    parser.add_argument(
        "--unfreeze_last_k", type=int, default=0, help="How many unfrozen blocks in case of fix backbone"
    )
    parser.add_argument(
                "--final_dropout", type=float, default=0.5, help="Dropout in the final layer"
                    )
    return parser

class TrainingModel(torch.nn.Module):

    def __init__(self, opt, subdir='.'):# subdir:first_test
        super(TrainingModel, self).__init__()

        # 为缺失属性提供默认值（与原 argparse 默认一致）
        defaults = {
            "lr": 0.0001,
            "optim": "adam",
            "fix_backbone": False,
            "no_cuda": False,
            "start_fresh": False,
            "use_leaky": False,
            "ckpt": None,
            "use_proj": False,
            "proj_ratio": None,
            "final_dropout": 0.5,
            "pretrain": None,
            "continue_epoch": None,
            "beta1": 0.9,
            "weight_decay": 0.0,
            "stay_positive": None,
            "unfreeze_last_k": 0,
            "use_contrastive": False,
            "batched_syncing": False,
        }
        # 给缺失属性赋默认值
        for key, val in defaults.items():
            if not hasattr(opt, key):
                setattr(opt, key, val)

        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, subdir) #./checkpointys/first_test
        self.device = torch.device('cpu') if opt.no_cuda else torch.device('cuda:0')

        #Granting functionality to start with random init instead of pretrained resnet50 weights
        # 允许功能：从随机初始化开始，而不是使用预训练的 ResNet50 权重”
        #self.model = create_architecture(opt.arch, pretrained=not opt.start_fresh,  num_classes=1,leaky=opt.use_leaky,ckpt=opt.ckpt, use_proj=self.opt.use_proj, proj_ratio=opt.proj_ratio,dropout=opt.final_dropout)
        self.model = create_architecture(opt.arch, pretrained=not opt.start_fresh, num_classes=1,
                                         ckpt=opt.ckpt, use_proj=self.opt.use_proj, proj_ratio=opt.proj_ratio,
                                         dropout=opt.final_dropout)
        # opt.arch：res50nodown；pretrained：是否加载预训练权重（pretrained）；是否加载自定义权重文件（ckpt）
        # num_classes=1：最后输出层的通道数。这里是1，表示是二分类任务（如真假图片）
        # leaky=opt.use_leaky：是否将 ReLU 激活函数替换为 leaky ReLU，防止“神经元死亡”问题
        # ckpt：路径字符串 或 None如果给了路径，就从指定 checkpoint 加载模型参数（而不是 ImageNet 权重）
        # use_proj是否使用“投影头（Projection Head）”——通常在对比学习中加入（如 SimCLR、SupCon）
        # proj_ratio：如 4。如果用了投影头，控制降维比例。例如原始 2048 维 -> 2048 / 4 = 512 维
        # dropout 如 0.5。设置最后一层的 Dropout 概率，防止过拟合。值为 0 表示不使用 Dropout
        num_parameters = count_parameters(self.model)# 统计模型 self.model 中所有可训练参数的总数量，即所有requires_grad=True的参数的元素个数之和。
        print(f"Arch: {opt.arch} with #trainable {num_parameters}")# 训练参数量的综合

        if hasattr(opt, "lr"):
            print('lr:', opt.lr)
        else:
            print('lr: (testing mode, not used)')
        # 它是二元交叉熵损失（Binary Cross Entropy, BCE） 和 Sigmoid 激活函数 的组合，专门用来计算模型输出与真实标签之间的误差。
        self.loss_fn = torch.nn.BCEWithLogitsLoss().to(self.device)

        if hasattr(self.opt, "fix_backbone") and self.opt.fix_backbone:# 4. 冻结骨干网络（可选）默认是0。如果设置了 --fix_backbone 参数：只微调最后几层，其余 ResNet 的 block 冻结还可以指定只解冻最后 k 个 block
            self.freeze_backbone(unfreeze_last_k=self.opt.unfreeze_last_k)
        # 根据用户选择（adam 或 sgd）创建优化器
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if opt.optim == "adam":
            self.optimizer = torch.optim.Adam(
                parameters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                parameters, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [adam, sgd]")
        # 可以加载预训练模型或上一次训练保存的中间模型。支持继续训练（--continue_epoch）
        if opt.pretrain:
            self.model.load_state_dict(
                torch.load(opt.pretrain, map_location="cpu")["model"]
            )
            print("opt.pretrain ", opt.pretrain)
        if opt.continue_epoch is not None:
            self.load_networks(opt.continue_epoch)
        self.model.to(self.device)


    def freeze_backbone(self, unfreeze_last_k: int = 0):
        """
        Freezes all backbone layers except for the last `k` blocks and the final fully connected (fc) layer.
        Additionally zero-initializes the weights and biases for layers not frozen.
        冻结所有主干（backbone）层，除了最后的 k 个 block 和最终的全连接（fc）层。
        此外，对于未被冻结的层，会将其权重和偏置初始化为零。
        Args:
            unfreeze_last_k (int): Number of blocks from the end to keep unfrozen.
            unfreeze_last_k（int）：从末尾开始保留未冻结的 block 数量。
        """
        # Get all blocks in the model (assuming blocks are named layer1, layer2, etc.)
        # 获取模型中的所有 block（假设这些 block 的命名为 layer1、layer2 等）。
        backbone_blocks = [name for name, _ in self.model.named_children() if name.startswith("layer")]
        
        # Determine the blocks to freeze and unfreeze
        # 确定要冻结和解冻的各个 block。
        blocks_to_unfreeze = backbone_blocks[-unfreeze_last_k:] if unfreeze_last_k > 0 else []
        for name, param in self.model.named_parameters():
            # Check if the parameter belongs to a backbone block
            # 检查该参数是否属于主干（backbone）中的某个 block。
            block_name = name.split('.')[0]
            if block_name in blocks_to_unfreeze or 'fc' in name:  # Unfreeze解冻
                param.requires_grad = True
                if 'fc' in name:
                    param.data.zero_()
            else:  
                param.requires_grad = False
                module = dict(self.model.named_modules())[block_name]
                module.eval()
    # 用于 调整学习率（learning rate）
    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:# 将优化器 self.optimizer 中所有参数组（param_groups）的学习率都缩小 10 倍。
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:# 断调整后的学习率是否低于给定的最小阈值 min_lr（默认1e-6）。返回是否继续训练的布尔值（True 表示还可以继续，False 表示学习率已经太低，不再继续）。
                return False
        return True
    # 这个函数 get_learning_rate 的作用是 获取当前优化器中第一个参数组的学习率。
    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]
    # 这个函数 train_on_batch 是 TrainingModel 类中用于训练模型单个批次数据的核心函数，负责完成一次前向传播、计算损失、反向传播和参数更新。
    def train_on_batch(self, data):
        self.total_steps += 1# 训练步数计数器加一。
        self.model.train() # 设置模型为训练模式（启用dropout、batchnorm等训练特性）。

        #grads = {name: [] for name, param in self.model.named_parameters() if param.requires_grad}
        if self.opt.batched_syncing:# 如果开启了 batched_syncing（可能用于同步真伪图像批次），
            rdata = data[0]
            fdata = data[1]
            input = torch.cat((rdata['img'], fdata['img']), dim=0).to(self.device)# 传入的数据是一个包含两个元素的tuple或list，分别是真数据 rdata 和伪数据 fdata。
            label = torch.cat((rdata['target'], fdata['target']), dim=0).to(self.device).float()# 把两者的图像和标签沿第0维（batch维）拼接成一个大batch，并移动到训练设备（CPU或GPU）。
        else:# 否则，直接取单个数据字典中的图像和标签，并移动到设备。
            input = data['img'].to(self.device)
            label = data['target'].to(self.device).float()
        output, feats = self.model(input, return_feats=self.opt.use_contrastive)# 将输入送入模型前向传播，输出结果 output 和中间特征 feats（如果 use_contrastive 为真，可能用于对比学习）。


        if len(output.shape) == 4:
            ss = output.shape
            loss = self.loss_fn(
                output,
                label[:, None, None, None].repeat(
                (1, int(ss[1]), int(ss[2]), int(ss[3]))
                ),
            )
        else:
            loss = self.loss_fn(output.squeeze(1), label)
        self.optimizer.zero_grad()# 清空上一轮梯度。
        loss.backward()# 反向传播计算当前梯度。
        self.optimizer.step()# 优化器根据梯度更新模型参数。

        # Stay-Positive Update (ICML)
        if self.opt.stay_positive == 'clamp':# 如果参数 stay_positive 设为 'clamp'，则对模型最后一层全连接层的权重执行非负约束（权重小于0的部分被置为0），可能是某篇ICML论文的技巧。
            with torch.no_grad():
                self.model.fc.weight.data.clamp_(min=0)
        return loss.cpu()

    def save_networks(self, epoch):
        save_filename = 'model_epoch_%s.pth' % epoch # 根据传入的 epoch（训练轮次）生成一个保存文件名，比如 "model_epoch_10.pth"。
        save_path = os.path.join(self.save_dir, save_filename)# 拼接完整的保存路径，通常 self.save_dir 是模型保存的目录。

        # serialize model and optimizer to dict
        state_dict = {
            'model': self.model.state_dict(),# 模型当前的权重参数，用 state_dict() 以字典形式返回（key是参数名，value是张量）。
            'optimizer': self.optimizer.state_dict(),# 优化器当前的状态（包括学习率、动量、历史梯度等），以便恢复训练时继续优化。
            'total_steps': self.total_steps,# 训练过程中累计的总步数（用于恢复训练进度）。
        }

        torch.save(state_dict, save_path) # 利用 PyTorch 的 torch.save 函数，将 state_dict 序列化保存到指定路径 save_path。
                                          # 该文件格式通常是 .pth，是 PyTorch 模型和状态的标准保存格式。

    # load models from the disk
    #这个 load_networks 函数的作用是：从磁盘加载模型的训练状态（包括模型权重、优化器状态、训练步数），用于断点恢复训练或验证已训练模型的性能。
    def load_networks(self, epoch):
        load_filename = 'model_epoch_%s.pth' % epoch # 构造要加载的模型文件名，比如：model_epoch_20.pth
        load_path = os.path.join(self.save_dir, load_filename)# 拼接为完整路径，例如：./checkpoints/first_test/model_epoch_20.pth

        print('loading the model from %s' % load_path)# 打印加载信息
        state_dict = torch.load(load_path, map_location=self.device)# 使用 torch.load 加载 .pth 文件，返回一个包含多个 key 的字典（即保存时的 state_dict）

        self.model.load_state_dict(state_dict['model'])# 恢复模型权重：将 state_dict['model'] 加载到当前模型结构中
        self.model.to(self.device)# 将模型移到正确的设备上（GPU 或 CPU）

        try:
            self.total_steps = state_dict['total_steps']# 如果保存的状态中有记录训练步数 total_steps，就恢复它；
        except:
            self.total_steps = 0# 否则默认为 0。

        #加载优化器状态字典。由于优化器状态中有可能保存了梯度或中间张量，这些需要手动 .to(self.device)，否则会因设备不匹配出错。如果失败（比如 optimizer 没有保存），就跳过。
        try:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        except:
            pass

    # 让模型对验证集（或测试集）的图片进行判断，输出它认为每张图是“真”还是“假”的可能性。
    def predict(self, data_loader):#这里传入的是一个 data_loader，里面装着很多图片和它们的标签，一批一批加载出来。
        model = self.model.eval()# 不训练，只预测让模型进入“预测”状态（会关闭 Dropout、BatchNorm 的训练行为）
        with torch.no_grad():# 告诉 PyTorch 不要计算梯度，这样能节省显存、加快速度（因为我们不训练）
            y_true, y_pred, y_path = [], [], []
            for data in tqdm.tqdm(data_loader):# 从 data_loader 里面一批批地拿出数据，每个 data 是一个字典。tqdm 是一个进度条工具，会在终端显示处理进度。
                img = data['img']#                     'img': 图片张量,
                label = data['target'].cpu().numpy()#  'target': 标签,
                paths = list(data['path'])          #  'path': 图片路径
                # 把标签转成 NumPy 数组，路径也转成列表，方便后面拼接。
                out_tens,_ = model(img.to(self.device))# 得到模型的输出 out_tens，这是预测的 raw 分数（比如对“真图”的信心程度）。
                out_tens = out_tens.cpu().numpy()[:, -1]# 把输出从 GPU 转回 CPU，并转换为 NumPy 数组。[:, -1]：取出最后一列的分数（一般是对“真图”的预测值）。
                assert label.shape == out_tens.shape# 防止出现“标签个数”和“预测值个数”不一致的问题。

                y_pred.extend(out_tens.tolist())# 把这批的预测分数、真实标签、路径都加到前面我们准备好的空列表中。
                y_true.extend(label.tolist())
                y_path.extend(paths)

        y_true, y_pred = np.array(y_true), np.array(y_pred)# 转成 NumPy 数组，方便后续评估指标的计算
        return y_true, y_pred, y_path
        #让模型对一批图片进行预测，并输出三样东西：
        # y_true：图片的真实标签（0 表示“假图”，1 表示“真图”）
        # y_pred：模型预测出来的分数（一般是大于 0 表示真，小于 0 表示假）
        # y_path：每张图片的文件路径（用于记录或分析）