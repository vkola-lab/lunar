# import math
import random
import collections.abc as collections
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import rotate
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    Resized,
    RandSpatialCropd,
    RandScaleCropd,
)

def transformsFuncd(type, image_size, keys, allow_missing_keys=True):
    if type == 'train':
        return Compose(
                [
                    LoadImaged(keys=keys, allow_missing_keys=allow_missing_keys),
                    EnsureChannelFirstd(keys=keys, allow_missing_keys=allow_missing_keys),
                    Spacingd(keys=keys, allow_missing_keys=True, pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                    monai.transforms.Orientationd(keys=keys, allow_missing_keys=True, axcodes="RAS"),
                    PadEqualSized(keys=keys, allow_missing_keys=True),
                    # Resized(keys=keys, allow_missing_keys=True, spatial_size=(image_size*2,)*3, mode='nearest'),
                    # RandSpatialCropd(keys=keys, allow_missing_keys=True, roi_size=(image_size,)*3, random_size=False, random_center=False),
                    RandScaleCropd(keys=keys, allow_missing_keys=True, roi_scale=0.4, max_roi_scale=1, random_size=True, random_center=True),
                    Resized(keys=keys, allow_missing_keys=True, spatial_size=(image_size,)*3),
                    FlipJitterd(keys=keys, allow_missing_keys=allow_missing_keys),
                    monai.transforms.RandGaussianSmoothd(keys=keys, prob=0.2, allow_missing_keys=allow_missing_keys),
                    MinMaxNormalized(keys=keys),
                ]            
            )
    else:
        return Compose(
            [
                LoadImaged(keys=keys, allow_missing_keys=allow_missing_keys),
                EnsureChannelFirstd(keys=keys, allow_missing_keys=allow_missing_keys),
                Spacingd(keys=keys, allow_missing_keys=allow_missing_keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                Orientationd(keys=keys, allow_missing_keys=allow_missing_keys, axcodes="RAS"),
                PadEqualSized(keys=keys, allow_missing_keys=allow_missing_keys),
                Resized(keys=keys, allow_missing_keys=allow_missing_keys, spatial_size=(image_size,)*3, mode='nearest'),
                MinMaxNormalized(keys=keys),
            ]
        )
    
# def transformsFunc(type, image_size):
#     if type == 'train':
#         return Compose(
#                 [
#                     monai.transforms.LoadImage(),
#                     monai.transforms.EnsureChannelFirst(),
#                     monai.transforms.Spacing(pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
#                     monai.transforms.CropForeground(),
#                     # monai.transforms.RandScaleCropd(keys=keys, roi_scale=0.4, max_roi_scale=1, random_size=True, random_center=True),
#                     monai.transforms.Resize(spatial_size=(image_size*2,)*3),
#                     monai.transforms.ResizeWithPadOrCrop(spatial_size=image_size),
#                     flip_and_jitter,
#                     monai.transforms.RandGaussianSmooth(prob=0.5),
#                     minmax_normalize,
#                 ]            
#             )
#     else:
#         return Compose(
#             [
#                 monai.transforms.LoadImage(),
#                 monai.transforms.EnsureChannelFirst(),
#                 monai.transforms.Spacing(pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
#                 monai.transforms.CropForeground(),
#                 monai.transforms.Resize(spatial_size=(image_size*2,)*3),
#                 monai.transforms.ResizeWithPadOrCrop(spatial_size=image_size),
#                 minmax_normalize,
#             ]
#         )


class SeededTransform:
    def __init__(self, transform, seed):
        self.transform = transform
        self.seed = seed

    def __call__(self, data):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        # ic(data)
        return self.transform(data)

class MinMaxNormalized:
    def __init__(self, keys=["image"]):
        self.keys = keys

    def __call__(self, x):
        for key in x:
            if key in self.keys:
                eps = torch.finfo(torch.float32).eps
                x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
        return x

class FlipJitterd:
    def __init__(self, keys, allow_missing_keys=True):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.transform = monai.transforms.Compose([
            monai.transforms.RandAxisFlipd(keys=keys, allow_missing_keys=True, prob=0.5),
            monai.transforms.RandRotated(keys=keys, allow_missing_keys=allow_missing_keys, prob=0.5, range_x=0.4),
            monai.transforms.RandBiasFieldd(keys=keys, allow_missing_keys=allow_missing_keys, prob=0.5),  # Random Bias Field artifact
            monai.transforms.RandGaussianNoised(keys=keys, allow_missing_keys=allow_missing_keys, prob=0.5),
        ])
    def __call__(self, x):
        return self.transform(x)


class PadEqualSized:
    def __init__(self, keys, allow_missing_keys=True, method='symmetric'):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.method = method
        self.cropforeground = monai.transforms.CropForeground()
        

    def __call__(self, x):
        for key in self.keys:
            if key in x:
                C, D, H, W = x[key].size()
                max_dim = max(D,H,W)

                pad_transform = Compose([
                    self.cropforeground,
                    monai.transforms.SpatialPad(spatial_size=(max_dim,)*3, method=self.method)
                ])
                x[key] = pad_transform(x[key])

        return x