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

import numbers
import os
import random
import sys
from io import BytesIO

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from .custom_aug import *
random.seed(17)
def make_processing(opt):
    # make precessing transform做图像预处理变换
    # input: an argparse.Namespace 输入：保存各种参数配置
    # output: a torchvision.transforms.Compose 输入：对象，包含一系列图像变换

    opt = parse_arguments(opt) # 把 opt 中的一些字符串形式的参数（例如 "bilinear,lanczos"、"0.0,3.0"）转换成Python 中可以使用的列表或数字形式，方便后续使用。因为刚开始传入的参数全都是字符串形式。
    transforms_list = list()  # 图像预处理的操作列表，把所有图像预处理过程打包

    transforms_pre = make_pre(opt)  # 做预处理数据增强变换
    if transforms_pre is not None:
        transforms_list.append(transforms_pre)

    transforms_aug = make_aug(opt) # 根据你的配置opt，构建一个包含“图像增强操作”的流程（transforms.Compose），用于数据增强（data augmentation）。
    if transforms_aug is not None:
        idx_aug = len(transforms_list)
        transforms_list.append(transforms_aug)
    else:
        idx_aug = -1

    transforms_post = make_post(opt)  # make post-data-augmentation transforms
    if transforms_post is not None:
        transforms_list.append(transforms_post)

    transforms_list.append(make_normalize(opt))  # make normalization #归一化

    if (hasattr(opt, "num_views")) and (abs(opt.num_views) > 0):#判断是否需要多视图增强（用于自监督学习）
        # hasattr(opt, "num_views")：检查 opt 是否有 num_views 这个参数；abs(opt.num_views) > 0：取绝对值，确保是非零；所以：如果用户设置了 num_views，并且值不为 0，就说明要进行多视图生成。
        print("num_view:", opt.num_views)
        t = transforms.Compose(transforms_list)# transforms.Compose(transforms_list)：把之前所有收集的增强操作合成一个 transform 流程；
        # make multiviews for Self-supervised learning (SSL)
        # 对一张图片重复进行 N 次不同的随机增强，生成多个“视图（view）”作为输入 —— 常见于对比学习（contrastive learning）等 SSL 方法中。
        t = MultiView([t for _ in range(abs(opt.num_views))])
    else:
        t = transforms.Compose(transforms_list)# 只构建单个 transform 流，不进行多视图复制。

    return t


def add_processing_arguments(parser):
    # parser is an argparse.ArgumentParser
    #
    # ICASSP2023: --cropSize 96 --loadSize -1 --resizeSize -1 --norm_type resnet --resize_prob 0.2 --jitter_prob 0.8 --colordist_prob 0.2 --cutout_prob 0.2 --noise_prob 0.2 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --hpf_prob 0.0 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100 --resize_size 256 --resize_ratio 0.75
    # ICME2021  : --cropSize 96 --loadSize -1 --resizeSize -1 --norm_type resnet --resize_prob 0.0 --jitter_prob 0.0 --colordist_prob 0.0 --cutout_prob 0.0 --noise_prob 0.0 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --hpf_prob 0.0 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100
    #

    parser.add_argument(
        "--resizeSize",
        type=int,
        default=-1,
        help="scale images to this size post augumentation",
    )
    parser.add_argument(
        "--loadSize",
        type=int,
        default=-1,
        help="scale images to this size pre augumentation",
    )
    parser.add_argument(
        "--cropSize",
        type=int,
        default=-1,
        help="crop images to this size post augumentation",
    )
    parser.add_argument("--no_random_crop", action="store_true")

    # data-augmentation probabilities
    parser.add_argument("--resize_prob", type=float, default=0.0)
    parser.add_argument("--jitter_prob", type=float, default=0.0)
    parser.add_argument("--colordist_prob", type=float, default=0.0)
    parser.add_argument("--cutout_prob", type=float, default=0.0)
    parser.add_argument("--noise_prob", type=float, default=0.0)
    parser.add_argument("--blur_prob", type=float, default=0.0)
    parser.add_argument("--cmp_prob", type=float, default=0.0)
    parser.add_argument("--rot90_prob", type=float, default=1.0)
    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument("--hpf_prob", type=float, default=0.0)
    parser.add_argument("--pre_crop_prob", type=float, default=0.0, help="Probability of applying a random crop to 256 before performing the random rz crop")

    # data-augmentation parameters
    parser.add_argument("--rz_interp", default="bilinear")
    parser.add_argument("--blur_sig", default="0.5")
    parser.add_argument("--cmp_method", default="cv2")
    parser.add_argument("--cmp_qual", default="75")
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--resize_ratio", type=float, default=1.0)

    # other
    parser.add_argument("--norm_type", type=str, default="resnet")  # normalization type
    # multi views for Self-supervised learning (SSL)
    parser.add_argument("--num_views", type=int, default=0)

    return parser


def parse_arguments(opt):
    # 判断第一个参数是不是第二个参数类型：eg:判断a是不是浮点型if not isinstance(a,float):
    """
        如果 rz_interp 还不是列表（说明是字符串），就用逗号分割成一个列表。
        比如："bilinear,lanczos" → ["bilinear", "lanczos"]
    """
    if not isinstance(opt.rz_interp, list):
        opt.rz_interp = list(opt.rz_interp.split(","))
    if not isinstance(opt.blur_sig, list):
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(",")]
    if not isinstance(opt.cmp_method, list):
        opt.cmp_method = list(opt.cmp_method.split(","))
    if not isinstance(opt.cmp_qual, list):
        opt.cmp_qual = [int(s) for s in opt.cmp_qual.split(",")]
        if len(opt.cmp_qual) == 2:
            opt.cmp_qual = list(range(opt.cmp_qual[0], opt.cmp_qual[1] + 1))
        elif len(opt.cmp_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --cmp_qual.")
    return opt


rz_dict = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}

pil_rz_dict = {
        "bilinear":Image.Resampling.BILINEAR,
        "bicubic":Image.Resampling.BICUBIC,
        "lanczos":Image.Resampling.LANCZOS,
        "nearest":Image.Resampling.NEAREST
        }

# 预处理
def make_pre(opt):
    transforms_list = list()
    if opt.loadSize > 0:
        print("\nUsing Pre Resizing\n")
        # 160-166行是为了支持用户传多个插值方式，程序从中随机选一个用于 resize，这样可以增加训练时的数据多样性（Data Augmentation）。
        transforms_list.append(
            transforms.Lambda(
                lambda img: TF.resize(
                    img,
                    opt.loadSize,
                    interpolation=rz_dict[sample_discrete(opt.rz_interp)],
                )
            )
        )
        #
        transforms_list.append(
            CenterCropPad(opt.loadSize, pad_if_needed=True, padding_mode="symmetric")
        )

    if len(transforms_list) == 0:
        return None
    else:
        return transforms.Compose(transforms_list)

# 后处理，详细的每块解释在OneNote中。
def make_post(opt):
    transforms_list = list()
    if opt.resizeSize > 0:# 后处理中的缩放
        print("\nUsing Post Resizing\n") # 后处理中的缩放
        transforms_list.append(
            transforms.Resize(
                opt.resizeSize, interpolation=transforms.InterpolationMode.BICUBIC
            )
        )
        transforms_list.append(transforms.CenterCrop((opt.resizeSize, opt.resizeSize)))

    if opt.cropSize > 0:#把图像裁剪成指定大小（cropSize），方式可以是随机裁剪，也可以是中心裁剪。
        if not opt.no_random_crop:
            print("\nUsing Post Random Crop\n")# 使用随即裁切
            transforms_list.append(
                transforms.RandomCrop(
                    opt.cropSize, pad_if_needed=True, padding_mode="symmetric"
                )
            )
        else:
            print("\nUsing Post Central Crop\n")# 使用中心裁切
            transforms_list.append(
                CenterCropPad(
                    opt.cropSize, pad_if_needed=True, padding_mode="symmetric"
                )
            )

    if len(transforms_list) == 0:
        return None
    else:
        return transforms.Compose(transforms_list)

# 增强 # 详细解释在OneNote
def make_aug(opt):
    # AUG
    transforms_list_aug = list()
    
    if opt.pre_crop_prob > 0:# 有一定概率对图像进行一次“前置随机裁剪”操作（裁成 256×256 的正方形）
        transforms_list_aug.append(transforms.RandomApply([transforms.RandomCrop(256, pad_if_needed=True, padding_mode="symmetric")],opt.pre_crop_prob))
    
    if (opt.resize_size > 0) and (opt.resize_prob > 0):  # 根据配置，对图像执行“随机缩放裁剪”（Random Resized Crop）增强操作，
        if opt.flex_rz:
            print('using flexible resize')
            transforms_list_aug.append(
                transforms.RandomApply(
                    [
                        RandomResizedCropWithVariableSize(
                                                min_size=128,
                                                max_size=1024,
                                                scale=(0.08, 1.0),
                                                ratio=(opt.resize_ratio, 1.0 / opt.resize_ratio),
                                                interpolation=rz_dict[sample_discrete(opt.rz_interp)]
                                            )
                        ],
                            opt.resize_prob,
                    )
                )
        else:
            transforms_list_aug.append(
                transforms.RandomApply(
                    [
                        transforms.RandomResizedCrop(
                            size=opt.resize_size,   # 最终图像缩放成的目标尺寸（正方形）
                            scale=(0.08, 1.0),      # 裁剪区域占原图面积的比例范围（最小 8%，最大 100%）
                            ratio=(opt.resize_ratio, 1.0 / opt.resize_ratio),# 裁剪区域的宽高比范围
                            interpolation=rz_dict[sample_discrete(opt.rz_interp)],# 使用的插值方式，比如双线性、立方等
                        )
                    ],
                    opt.resize_prob,
                )
            )
    # 让图像在颜色上更丰富多变，增强模型的泛化能力。
    if opt.jitter_prob > 0:# 以一定概率（opt.jitter_prob）对图像进行颜色抖动（亮度、对比度、饱和度、色相的扰动），以增强图像的颜色多样性，让模型不依赖特定的颜色分布。
        transforms_list_aug.append(
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=opt.jitter_prob
            )
        )

    if opt.colordist_prob > 0:# 1.灰度处理（降低颜色信息）
        transforms_list_aug.append(transforms.RandomGrayscale(p=opt.colordist_prob))#以 opt.colordist_prob 的概率，把图像转换成灰度图（只有黑白信息），目的是让模型不要太依赖颜色。



    if opt.cutout_prob > 0:#  2. Cutout（随机遮挡）
        transforms_list_aug.append(create_cutout_transforms(opt.cutout_prob))# 调用函数生成一个 cutout 操作：在图像上随机盖一个灰色/中性色的方块区域。增强模型的遮挡鲁棒性（比如人脸的一部分被遮住也能识别）。

    if opt.noise_prob > 0:# 3. 加噪声
        transforms_list_aug.append(create_noise_transforms(opt.noise_prob))# 给图像加一些高斯噪声，模拟图像压缩或拍照时的噪点。帮助模型适应更“脏”的输入图像。

    if opt.blur_prob > 0:# 4. 模糊处理（高斯模糊）
        transforms_list_aug.append(
            transforms.Lambda(
                lambda img: data_augment_blur(img, opt.blur_prob, opt.blur_sig)#使用高斯模糊模拟失焦模糊、运动模糊。blur_sig 控制模糊强度；blur_prob 控制概率。
            )
        )

    if opt.cmp_prob > 0:# 5.  图像压缩模拟（JPEG/WebP）
        transforms_list_aug.append(
            transforms.Lambda(
                lambda img: data_augment_cmp(
                    img, opt.cmp_prob, opt.cmp_method, opt.cmp_qual# 模拟图像压缩后的伪影/失真（比如发到微信、保存为 .jpg 后的效果）。
                )#opt.cmp_prob （例如 0.5）压缩增强的概率，值为 0～1。表示以多大概率对当前图像执行压缩（其余时候图像不变）。举例：0.5 表示 50% 概率对图像执行 JPEG/WebP 压缩。
                 #opt.cmp_method （例如 ['jpeg', 'webp']）压缩的方法/格式，可以是："jpeg"：经典的 JPEG 压缩格式"webp"：较新、更高效的压缩格式实际运行时一般会从这个列表中随机选择一种方法来对图像进行压缩。
                 #opt.cmp_qual （例如 [30, 70, 90]）压缩质量参数，越低的数值 → 压缩越严重 → 图像伪影越多。通常在 10~100 之间，100 表示最高质量（几乎无损）。函数内部会随机选一个压缩质量数值，对当前图像进行压缩模拟。



            )
        )

    if opt.rot90_prob > 0:# 6. 随机旋转 90 度的倍数[0,90,180,270]
        transforms_list_aug.append(
            transforms.Lambda(lambda img: data_augment_rot90(img, opt.rot90_prob))# 随机将图像旋转 0、90、180 或 270 度。让模型学习到不受方向影响的特征（比如鸟是横的、竖的都能识别）。
        )# opt.rot90_prob：表示进行 90 度旋转增强的概率，取值范围是 0～1。

    if opt.hpf_prob > 0:# 7. 高通滤波（保留边缘细节）
        transforms_list_aug.append(transforms.ToTensor())
        transforms_list_aug.append(
            transforms.Lambda(
                lambda img: data_augment_hpf(img, opt.hpf_prob, opt.blur_sig)# 使用高通滤波器强化图像的“高频信息”，比如边缘、细节、纹理。
            )# opt.hpf_prob 高通滤波操作的执行概率，是一个 0～1 的浮点数；
             # opt.blur_sig 表示 高斯模糊的 sigma（标准差）范围；是一个列表，比如 [0.5, 1.0, 2.0]，内部通常会随机取一个值来做 模糊操作；
        )# 表示：以一定概率 opt.hpf_prob，对输入图像 img 施加“高通滤波”操作（即提取图像的细节/边缘部分），其模糊程度由 opt.blur_sig 控制。

    if not opt.no_flip:# 8. 随机水平翻转# 真变假，假变真
        transforms_list_aug.append(transforms.RandomHorizontalFlip())# 左右翻转图像。经典的数据增强手段，不会改变语义却能增加数据多样性。

    if len(transforms_list_aug) > 0:# 9. 返回最终的变换组合
        return transforms.Compose(transforms_list_aug)#把所有的增强操作组合成一个流程（用 Compose）。
    else:
        return None# 如果什么都没添加，就返回 None（代表不使用增强）。


def make_normalize(opt):
    transforms_list = list()

    if opt.norm_type == "resnet":
        print("normalize RESNET")

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    elif opt.norm_type == "clip":
        print("normalize CLIP")
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
        )
    elif opt.norm_type == "xception":
        print("normalize -1,1")

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )
    elif opt.norm_type == "spec":
        print("normalize SPEC")

        transforms_list.append(normalization_fft)
        transforms_list.append(transforms.ToTensor())

    elif opt.norm_type == "fft2":
        print("normalize Energy")

        transforms_list.append(pic2imgn)
        transforms_list.append(normalization_fft2)
        transforms_list.append(imgn2torch)

    elif opt.norm_type == "residue3":
        print("normalize Residue3")

        transforms_list.append(normalization_residue3)
    elif opt.norm_type == "npr":
        print("normalize NPR")

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        from torch.nn.functional import interpolate
        transforms_list.append(
            lambda x: x[..., :(x.shape[-2]//2*2), :(x.shape[-1]//2*2)]
        )
        transforms_list.append(
            lambda x: (x - interpolate(x[None,...,::2,::2], scale_factor=2.0, mode='nearest', recompute_scale_factor=True)[0])*2.0/3.0
        )
    elif opt.norm_type == "cooc":
        print("normalize COOC")

        transforms_list.append(normalization_cooc)
    else:
        assert False

    return transforms.Compose(transforms_list)


class MultiView:
    def __init__(self, trasfroms_list):
        self.trasfroms_list = trasfroms_list
        print("num_view:", len(self.trasfroms_list))

    def __call__(self, x):
        return torch.stack([fun(x) for fun in self.trasfroms_list], 0)


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def data_augment_blur(img, p, blur_sig):
    if random.random() < p:
        img = np.array(img)
        sig = sample_continuous(blur_sig)
        gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sig)
        gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sig)
        gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sig)
        img = Image.fromarray(img)

    return img


def data_augment_hpf(img, p, blur_sig):
    assert isinstance(img, torch.Tensor)
    if random.random() < p:
        sig = 0.4 + sample_continuous(blur_sig)
        kernel_size = int(7 * sig)
        kernel_size = kernel_size + (kernel_size + 1) % 2
        img = img - TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sig)
        img = img + torch.from_numpy(np.asarray([[[0.485]], [[0.456]], [[0.406]]]))
    return img.float()


def data_augment_cmp(img, p, cmp_method, cmp_qual):
    if random.random() < p:
        img = np.array(img)
        method = sample_discrete(cmp_method)
        qual = sample_discrete(cmp_qual)
        img = cmp_from_key(img, qual, method)
        img = Image.fromarray(img)

    return img


def data_augment_rot90(img, p):
    if random.random() < p:
        angle = sample_discrete([0, 90, 180, 270])
        img = img.rotate(angle, expand=True)

    return img


def data_augment_D4(img, p):
    if random.random() < p:
        angle = sample_discrete([0, 90, 180, 270])
        sim = sample_discrete([0, 1])
        img = img.rotate(angle, expand=True)
        if sim == 1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img



def create_noise_transforms(p, var_limit=(10.0, 50.0)):
    from albumentations.augmentations.transforms import GaussNoise

    aug = GaussNoise(var_limit=var_limit, always_apply=False, p=p)
    return transforms.Lambda(
        lambda img: Image.fromarray(aug(image=np.array(img))["image"])
    )


def create_cutout_transforms(p):
    try:
        from albumentations import CoarseDropout
    except:
        from albumentations import CoarseDropout
    aug = CoarseDropout(
        max_holes=8,
        max_height=8,
        max_width=8,
        fill_value=128,
        always_apply=False,
        p=p,
    )
    return transforms.Lambda(
        lambda img: Image.fromarray(aug(image=np.array(img))["image"])
    )


def cv2_webp(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), compress_val]
    result, encimg = cv2.imencode(".webp", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_webp(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="webp", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


# NOTE: 'cv2' and 'pil' have been left here for legacy reasons
cmp_dict = {
    "cv2": cv2_jpg,
    "cv2_jpg": cv2_jpg,
    "cv2_webp": cv2_webp,
    "pil": pil_jpg,
    "pil_jpg": pil_jpg,
    "pil_webp": pil_webp,
}


def cmp_from_key(img, compress_val, key):
    return cmp_dict[key](img, compress_val)


def pic2imgn(pic):
    from copy import deepcopy

    img = np.float32(deepcopy(np.asarray(pic))) / 256.0
    return img


def imgn2torch(img):
    return torch.from_numpy(img).permute(2, 0, 1).float().contiguous()


def normalization_fft2(img, normalize=512.0):
    img = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)), axes=(0, 1))
    img = np.square(np.abs(img)) / normalize
    return img


def normalization_fft(pic):
    from copy import deepcopy

    im = np.float32(deepcopy(np.asarray(pic))) / 255.0

    for i in range(im.shape[2]):
        img = im[:, :, i]
        fft_img = np.fft.fft2(img)
        fft_img = np.log(np.abs(fft_img) + 1e-3)
        fft_min = np.percentile(fft_img, 5)
        fft_max = np.percentile(fft_img, 95)
        if (fft_max - fft_min) <= 0:
            print("ma cosa...")
            fft_img = (fft_img - fft_min) / ((fft_max - fft_min) + np.finfo(float).eps)
        else:
            fft_img = (fft_img - fft_min) / (fft_max - fft_min)
        fft_img = (fft_img - 0.5) * 2
        fft_img[fft_img < -1] = -1
        fft_img[fft_img > 1] = 1
        im[:, :, i] = fft_img

    return im


def normalization_residue3(pic, flag_tanh=False):
    from copy import deepcopy

    x = np.float32(deepcopy(np.asarray(pic))) / 32
    wV = (
        -1 * x[1:-3, 2:-2, :]
        + 3 * x[2:-2, 2:-2, :]
        - 3 * x[3:-1, 2:-2, :]
        + 1 * x[4:, 2:-2, :]
    )
    wH = (
        -1 * x[2:-2, 1:-3, :]
        + 3 * x[2:-2, 2:-2, :]
        - 3 * x[2:-2, 3:-1, :]
        + 1 * x[2:-2, 4:, :]
    )
    ress = np.concatenate((wV, wH), -1)
    if flag_tanh:
        ress = np.tanh(ress)

    ress = torch.from_numpy(ress).permute(2, 0, 1).contiguous()

    return ress


def normalization_cooc(pic):
    from copy import deepcopy

    x = deepcopy(np.asarray(pic))
    y = x[1:, 1:, :]
    x = x[:-1, :-1, :]
    bins = np.arange(257)
    H = np.stack(
        [
            np.histogram2d(
                x[:, :, i].flatten(), y[:, :, i].flatten(), bins, density=True
            )[0]
            for i in range(x.shape[2])
        ],
        0,
    )
    H = torch.from_numpy(H)
    return H


class CenterCropPad:
    def __init__(
        self, siz, pad_if_needed=False, padding_fill=0, padding_mode="constant"
    ):
        if isinstance(siz, numbers.Number):
            siz = (int(siz), int(siz))
        self.siz = siz
        self.pad_if_needed = pad_if_needed
        self.padding_fill = padding_fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        crop_height, crop_width = self.siz
        image_width, image_height = img.size[1], img.size[0]
        crop_top = (image_height - crop_height) // 2
        crop_left = (image_width - crop_width) // 2
        if crop_top < 0:
            if self.pad_if_needed:
                img = TF.pad(
                    img,
                    (0, -crop_top, 0, crop_height - image_height + crop_top),
                    fill=self.padding_fill,
                    padding_mode=self.padding_mode,
                )
            else:
                crop_height = image_height
            crop_top = 0
        if crop_left < 0:
            if self.pad_if_needed:
                img = TF.pad(
                    img,
                    (-crop_left, 0, crop_width - image_width + crop_left, 0),
                    fill=self.padding_fill,
                    padding_mode=self.padding_mode,
                )
            else:
                crop_width = image_width
            crop_left = 0
        return img.crop(
            (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
        )
