# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


from pytorch.common.image_preprocessing.cutout import Cutout


class TorchImageProcessor:
    """Simple data processors"""

    def __init__(self, image_size, is_color, mean, scale,
                 crop_size=0, pad=28,extend_size=300, color='BGR',
                 use_cutout=False,
                 use_mirroring=False,
                 use_random_crop=False,
                 use_center_crop=False,
                 use_random_gray=False):
        """Everything that we need to init"""
        torch.set_num_threads(1)

        self.image_size = image_size
        self.pad = pad
        self.mean = mean
        self.scale = scale
        self.crop_size = crop_size
        self.extend_size = extend_size
        self.color = color
        self.use_cutout = use_cutout
        self.use_mirroring = use_mirroring
        self.use_random_crop = use_random_crop
        self.use_center_crop = use_center_crop
        self.use_random_gray = use_random_gray
        self.save = True

    def process(self, image_path):
        """
        Returns processed data.
        """
        try:
            image = cv2.imread(image_path)
        except:
            image = image_path

        if image is None:
            print(image_path)

        # TODO: реализуйте процедуры аугментации изображений используя OpenCV и TorchVision
        # на выходе функции ожидается массив numpy с нормированными значениям пикселей

        to_plt = transforms.ToPILImage()
        image = to_plt(image)

        # Padder = transforms.Pad(self.pad)
        # image = Padder(image)

        if self.use_cutout:
            image = Cutout(image)
        if self.use_mirroring:
            fliper = transforms.RandomHorizontalFlip()
            image = fliper(image)
        if self.use_random_crop:
            RandomCrop = transforms.RandomCrop(self.crop_size)
            image = RandomCrop(image)
        if self.use_center_crop:
            CenterCrop = transforms.CenterCrop(self.crop_size)
            image = CenterCrop(image)
        if self.use_random_gray:
            gs = transforms.RandomGrayscale()
            image = gs(image)

        resizer = transforms.Resize(self.extend_size)
        image = resizer(image)

        RandomCrop = transforms.RandomCrop(self.image_size)
        image = RandomCrop(image)
        # if self.save:
        #     image.save('temp.png')
        #     print (123)
        #     self.save = False

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)

        # normalize = transforms.Normalize(self.mean, self.scale)
        # image = normalize(image)

        return image.numpy()
