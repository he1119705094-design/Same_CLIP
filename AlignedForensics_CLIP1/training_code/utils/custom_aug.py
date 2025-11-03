import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import random
import math
import numbers
import sys
import warnings
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage
from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None


def resized_crop(
    img: Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: Optional[bool] = True,
) -> Tensor:
    """Crop the given image and resize it to desired size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.NEAREST_EXACT``, ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are
            supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True`` (default): will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The default value changed from ``None`` to ``True`` in
            v0.17, for the PIL and Tensor backends to be consistent.
    Returns:
        PIL Image or Tensor: Cropped image.
    """
    img = transforms.functional.crop(img, top, left, height, width)
    img = transforms.functional.resize(img, size, interpolation, antialias=antialias)
    return img

class RandomResizedCropWithVariableSize(transforms.RandomResizedCrop):
    def __init__(self, min_size, max_size, scale=(0.08, 1.0), ratio=(1.0, 1.0), interpolation=transforms.InterpolationMode.BILINEAR):
        self.min_size = min_size
        self.max_size = max_size
        super().__init__(size=min_size, scale=scale, ratio=ratio, interpolation=interpolation)
    
    def get_random_size(self):
        """Return a random size between min_size and max_size."""
        size = random.randint(self.min_size, self.max_size)
        return size

    def __call__(self, img):
        size = img.size 
        size = tuple(int(element * 0.54) for element in size)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        ret =  resized_crop(img, i, j, h, w, size, self.interpolation, antialias=self.antialias)
        return ret
