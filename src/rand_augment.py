import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw

import numpy as np

from PIL import Image
from torchvision.transforms import transforms


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Rotate(img, magnitude):  # [-30, 30]
    assert -30 <= magnitude <= 30

    if random.random() > 0.5:
        magnitude = -magnitude

    return img.rotate(magnitude)


def Posterize(img, magnitude):  # [4, 8]
    assert 4 <= magnitude <= 8

    magnitude = int(magnitude)

    return PIL.ImageOps.posterize(img, magnitude)


def Solarize(img, magnitude):  # [0, 256]
    assert 0 <= magnitude <= 256

    return PIL.ImageOps.solarize(img, magnitude)


def Color(img, magnitude):  # [0.1, 1.9]
    assert 0.1 <= magnitude <= 1.9

    return PIL.ImageEnhance.Color(img).enhance(magnitude)


def Contrast(img, magnitude):  # [0.1, 1.9]
    assert 0.1 <= magnitude <= 1.9

    return PIL.ImageEnhance.Contrast(img).enhance(magnitude)


def Brightness(img, magnitude):  # [0.1, 1.9]
    assert 0.1 <= magnitude <= 1.9

    return PIL.ImageEnhance.Brightness(img).enhance(magnitude)


def Sharpness(img, magnitude):  # [0.1, 1.9]
    assert 0.1 <= magnitude <= 1.9

    return PIL.ImageEnhance.Sharpness(img).enhance(magnitude)


def ShearX(img, magnitude):  # [-0.3, 0.3]
    assert -0.3 <= magnitude <= 0.3

    if random.random() > 0.5:
        magnitude = -magnitude

    return img.transform(img.size, PIL.Image.AFFINE, (1, magnitude, 0, 0, 1, 0))


def ShearY(img, magnitude):  # [-0.3, 0.3]
    assert -0.3 <= magnitude <= 0.3

    if random.random() > 0.5:
        magnitude = -magnitude

    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, magnitude, 1, 0))


def TranslateX(img, magnitude):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= magnitude <= 0.45

    if random.random() > 0.5:
        magnitude = -magnitude
    magnitude = magnitude * img.size[0]

    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, magnitude, 0, 1, 0))


def TranslateY(img, magnitude):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= magnitude <= 0.45

    if random.random() > 0.5:
        magnitude = -magnitude
    magnitude = magnitude * img.size[1]

    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, magnitude))


def Cutout(img, magnitude):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= magnitude <= 0.2

    if magnitude <= 0.:
        return img

    magnitude = magnitude * img.size[0]

    return CutoutAbs(img, magnitude)


def CutoutAbs(img, magnitude):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= magnitude <= 20
    if magnitude < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - magnitude / 2.))
    y0 = int(max(0, y0 - magnitude / 2.))
    x1 = min(w, x0 + magnitude)
    y1 = min(h, y0 + magnitude)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)

    return img


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)

    return PIL.ImageOps.solarize(img, threshold)


def RandomHorizontalFlip(img, magnitude):
    if np.random.rand(1) < magnitude:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def RandomVerticalFlip(img, magnitude):
    if np.random.rand(1) < magnitude:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def RandomResizedCropWrap(img, magnitude):
    tfms = transforms.RandomResizedCrop(size=256, scale=(0.7, 0.7 + magnitude))
    return tfms(img)


def ColorJitter(img, _):
    tfms = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
    return tfms(img)


def augmentation_list():
    augmentation_list = [

        (Rotate, -30, 30),  # 4
        (ColorJitter, 0, 0),  # 7
        (Cutout, 0, 0.2),  # 15
        (RandomHorizontalFlip, 0, 1),
        (RandomVerticalFlip, 0, 1),
        (RandomResizedCropWrap, 0.0, 0.3),

        (Brightness, 1.0, 1.5),  # 9
        (Sharpness, 1.0, 1.5),  # 10
        (ShearX, -0.1, 0.1),  # 11
        (ShearY, -0.1, 0.1),  # 12
        (TranslateX, -0.1, 0.1),  # 13
        (TranslateY, -0.1, 0.1),  # 14

        # (AutoContrast, 0, 1),  # 1
        # (Equalize, 0, 1),  # 2
        # (Invert, 0, 1),  # 3
        # (Posterize, 4, 8),  # 5
        # (Contrast, 0.1, 1.9),  # 8
        # (SolarizeAdd, 0, 110)  # 16
        # (Solarize, 0, 256),  # 6
    ]

    return augmentation_list


class RandAugment:
    def __init__(self, params):
        """Generate a set of distortions.
        Args:
        params: dictionary for RandAugment parameters
        """
        self.N = params['N']
        self.M = params['M']  # [0, 30]
        seed = params['seed']
        random.seed(seed)
        self.augment_list = augmentation_list()
        self.augments = random.sample(self.augment_list, k=self.N)
        self.magnitudes = [np.random.randint(0, self.M + 1) for _ in range(len(self.augments))]

    def __call__(self, img, *args, **kwargs):
        for aug_params, mag in zip(self.augments, self.magnitudes):
            policy, low, high = aug_params
            # converting magnitude to the scale of augmentation
            magnitude = mag / 30. * float(high - low) + low
            img = policy(img, magnitude)

        return img
