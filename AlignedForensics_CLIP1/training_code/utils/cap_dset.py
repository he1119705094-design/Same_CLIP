import os
import pprint
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import random
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import transforms
from .record_transform import *
import time
import glob

def get_fakes(root_dir):
    image_list = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '1_fake' in dirnames:
            b_dir = os.path.join(dirpath, '1_fake')
            images = glob.glob(os.path.join(b_dir, '*.*'))
            image_list.extend(images)

    return image_list

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)



class CapDataset(Dataset):
    def __init__(self, root_dir, data_cap,transform=None,batched_syncing=False,use_inversions=False,seed=17):
        self.root_dir = root_dir
        self.transform = transform
        self.batched_syncing = batched_syncing
        paths = os.listdir(os.path.join(self.root_dir, '0_real'))
        if self.batched_syncing:
            self.recorder = TransformRecorder(self.transform)
        self.files = []
        random.seed(seed)
        paths = sorted(paths)
        if data_cap is not None:
            paths = random.sample(paths, data_cap)

        # if use_inversions:
        #     for path in paths:
        #         rpath = os.path.join(os.path.join(self.root_dir, '0_real'), path)
        #         fpath = os.path.join(os.path.join(self.root_dir, '1_fake'),path.replace('.jpg','.png'))
        #         self.files.append((rpath, 0))
        #         if not self.batched_syncing:
        #             self.files.append((fpath,1))
        if use_inversions:
            for path in paths:
                rpath = os.path.join(self.root_dir, '0_real', path)
                fpath = os.path.join(self.root_dir, '1_fake', path.replace('.jpg', '.png'))

                if os.path.isfile(rpath):
                    self.files.append((rpath, 0))
                else:
                    print(f"⚠️ 缺失真实图片: {rpath}")

                if not self.batched_syncing:
                    # 若 .png 不存在，试着换成 .jpg
                    if not os.path.isfile(fpath):
                        fpath = fpath.replace('.png', '.jpg')
                    if os.path.isfile(fpath):
                        self.files.append((fpath, 1))
                    else:
                        print(f"⚠️ 缺失伪造图片: {fpath}")
        else:
            fpaths = get_fakes(root_dir=self.root_dir)
            fpaths = [f for f in fpaths if os.path.isfile(f)]
            fpaths = random.sample(fpaths, min(data_cap, len(fpaths)))

            for idx, path in enumerate(paths[:len(fpaths)]):  # 保证一一对应
                rpath = os.path.join(self.root_dir, '0_real', path)
                fpath = fpaths[idx]

                if os.path.isfile(rpath):
                    self.files.append((rpath, 0))
                else:
                    print(f"⚠️ 缺失真实图像: {rpath}")

                if os.path.isfile(fpath):
                    self.files.append((fpath, 1))
                else:
                    print(f"⚠️ 缺失伪造图像: {fpath}")
            #
            # fpaths = get_fakes(root_dir=self.root_dir)
            # fpaths = random.sample(fpaths, data_cap)
            # for idx, path in enumerate(paths):
            #     rpath = os.path.join(os.path.join(self.root_dir, '0_real'), path)
            #     fpath = os.path.join(self.root_dir, fpaths[idx])
            #     self.files.append((rpath, 0))
            #     self.files.append((fpath,1))
            #fpaths = os.listdir(os.path.join(self.root_dir, '1_fake'))


    def filter_dataset(self, keep_count):
        self.files = self.files[keep_count:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path, target = self.files[index]
        sample = default_loader(path)#Image.open(path)
        if self.batched_syncing:
            SEED = int(time.time())
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            #torch.manual_seed(current_time_seed)
            sample = self.transform(sample)
            fpath = path.replace('.jpg', '.png').replace('0_real', '1_fake')
            if not os.path.isfile(fpath):
                alt_fpath = fpath.replace('.png', '.jpg')
                if os.path.isfile(alt_fpath):
                    fpath = alt_fpath
                else:
                    print(f"⚠️ Batched-sync 模式下找不到伪造图像: {fpath}，已跳过索引 {index}")
                    return self.__getitem__((index + 1) % len(self))  # 递归获取下一个样本

            fsample = default_loader(fpath)
            #
            # fpath = path.replace('.jpg','.png').replace('0_real', '1_fake')
            # if not os.path.exists(fpath):
            #     fpath = fpath.replace('.png', '.jpg')
            # fsample = default_loader(fpath)
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            fsample = self.transform(fsample)
            return {"img": sample, "target": target, "path": path}, {"img": fsample, "target": 1, "path": fpath}
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            return {"img": sample, "target": target, "path": path}

