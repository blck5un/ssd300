import random
import numpy as np
from PIL import Image
from torchvision import transforms


class Compose:
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None):
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.t = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image, boxes=None):
        image = self.t(image)
        return image, boxes


class ToTensor:
    def __init__(self):
        self.t = transforms.ToTensor()

    def __call__(self, image, boxes=None):
        image = self.t(image)
        return image, boxes


class Resize:
    def __init__(self, shape=(300, 300)):
        self.shape = shape
        self.t = transforms.Resize(self.shape)

    def __call__(self, image, boxes=None):
        image = self.t(image)
        return image, boxes


class RandomMirror:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = boxes.clone()
            boxes[:, 0::2] = 1 - boxes[:, 0::2].flip(1)
        return image, boxes


class RandomAffine:
    def __init__(self, x_range=(-0.3, 0.3), y_range=(-0.3, 0.3), p=0.5):
        self.p = p
        self.x_range = x_range
        self.y_range = y_range

    def __call__(self, image, boxes):
        if random.random() < self.p:
            # генерируем приращения координат x и y в относительных координатах (от 0 до 1)
            dx = np.random.uniform(*self.x_range)
            dy = np.random.uniform(*self.y_range)
            # сдвигаем картинку
            w, h = image.size
            image = image.transform(size=image.size, method=Image.AFFINE, data=((1, 0, dx * w, 0, 1, dy * h)))
            # сдвигаем боксы
            boxes = boxes.clone()
            boxes[:, 0::2] = (boxes[:, 0::2] - dx).clip(0, 1)
            boxes[:, 1::2] = (boxes[:, 1::2] - dy).clip(0, 1)
        return image, boxes


class RandomScale:
    def __init__(self, scale_range=(0.5, 1.2), p=0.5):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, image, boxes):
        if random.random() < self.p:
            scale = np.random.uniform(*self.scale_range)
            # считаем координаты картинки, которые после масштабирования совместятся с углами кадра
            # при масштабировании центр картинки считаем неподвижной точкой
            w, h = image.size
            coef = 1 / scale
            x0 = 0.5 * w * (1 - coef)
            y0 = 0.5 * h * (1 - coef)
            x1 = w * coef + x0
            y1 = h * coef + y0
            image = image.transform(size=image.size, method=Image.EXTENT, data=(x0, y0, x1, y1))

            boxes = boxes.clone()
            # переходим к координатам относительно центра картинки (неподвижная точка)
            boxes = boxes - 0.5
            # масштабируем
            boxes = boxes * scale
            # возвращаемя к координатам относительно (0, 0)
            boxes = (boxes + 0.5).clip(0, 1)
        return image, boxes


class RandomRotation:
    def __init__(self, angle_range=(-10, 10), p=0.5):
        self.p = p
        self.angle_range = angle_range

    def __call__(self, image, boxes):
        if random.random() < self.p:
            angle = np.random.uniform(*self.angle_range)
            image = image.rotate(angle=angle)
        # не вращаем боксы (приемлемо при малых углах вращения картинки)
        # требуется доработка
        return image, boxes


class RandomBlur:
    def __init__(self, p=0.5, kernel_size=11, sigma=(0.1, 5)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.t = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)

    def __call__(self, image, boxes):
        if random.random() < self.p:
            image = self.t(image)
        return image, boxes


class RandomColorJitter:
    def __init__(self, p=0.5, brightness=(0.3, 1.0), contrast=(0.8, 1.2)):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.t = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast)

    def __call__(self, image, boxes):
        if random.random() < self.p:
            image = self.t(image)
        return image, boxes
