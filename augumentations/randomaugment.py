# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg): # Maximize (normalize) image contrast.
    return PIL.ImageOps.autocontrast(img)

def Brightness(img, v, max_v, bias=0):  # Adjust image brightness.
    v = _float_parameter(v, max_v) + bias
    # print(v)
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):   # Adjust image color balance
    v = _float_parameter(v, max_v) + bias
    # print(v)
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):    # Adjust image contrast.
    v = _float_parameter(v, max_v) + bias
    # print(v)
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):     # Equalize the image histogram
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):   # Invert (negate) the image.
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):   # Reduce the number of bits for each color channel.
    v = _int_parameter(v, max_v) + bias
    # print(v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):  #  for rotating an image
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):   # Adjust image sharpness
    v = _float_parameter(v, max_v) + bias
    # print("v: ", v)
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):  #Transforms this image. This method creates a new image with the given size, and the same mode as the original, and copies data to the new image using the given transform.
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
# Transform.EXTENT (cut out a rectangular subregion), Transform.AFFINE (affine transform), Transform.PERSPECTIVE (perspective transform), Transform.QUAD (map a quadrilateral to a rectangle), or Transform.MESH (map a number of source quadrilaterals in one operation)

def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):    # Invert all pixel values above a threshold.
    v = _int_parameter(v, max_v) + bias
    # print(v)
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [# (AutoContrast, None, None),
            (Brightness, 0.5, 0.5), # not suitable for depth estimation
            # (Color, 0.9, 0.05),
            # (Contrast, 0.5, 0.2),
            # (Equalize, None, None), # not work
            # (Identity, None, None),
            (Posterize, 4, 4),
            # (Rotate, 30, 0), # not suitable for depth estimation
            # (Sharpness, 0.9, 0.05),
            # (ShearX, 0.3, 0),   # not suitable for depth estimation
            # (ShearY, 0.3, 0),   # not suitable for depth estimation
            # (Solarize, 256, 20),
            # (TranslateX, 0.3, 0), # not suitable for depth estimation
            # (TranslateY, 0.3, 0)   # not suitable for depth estimation
            ]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        # ops = random.choices(self.augment_pool, k=self.n)
        # print(ops)
        ops = self.augment_pool
        # print(ops)
        for op, max_v, bias in ops:
            # print(op)
            # print(max_v)
            # print(bias)
            # v = np.random.rand(1) # Random values in a given shape.
            # print("v: ", v)
            # if random.random() < 0.99:   # Return random number between 0.0 and 1.0:
            img = op(img, 0.5, max_v, bias)
        # img = CutoutAbs(img, int(32*0.5))
        return img