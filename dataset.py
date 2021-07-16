import os
import numpy as np
import albumentations as A
from config import cfg
from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class AnimeSketchColorDataset(Dataset):
    """
    Anime sketch dataset for colorization. Link (https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair)
    """
    def __init__(self, img_folder):
        self.img_folder = img_folder
        self.img_names = [os.path.join(img_folder, n) for n in os.listdir(img_folder)
                          if n.endswith(('png', 'jpeg', 'jpg'))]

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        img = np.array(Image.open(img_path))

        input_image = img[:, :512, :]
        target_image = img[:, 512:, :]

        augmentation = self._transform_both(image=input_image, img_target=target_image)
        input_image = augmentation['image']
        target_image = augmentation['img_target']

        input_image = self._transform_input_img(image=input_image)['image']
        target_image = self._transform_only_target(image=target_image)['image']

        return input_image, target_image

    def __len__(self):
        return len(self.img_names)

    @property
    def _transform_both(self):
        """Transforms both image"""
        return A.Compose(
            [A.Resize(width=cfg.IMG_SIZE, height=cfg.IMG_SIZE), ], additional_targets={"img_target": "image"}
        )

    @property
    def _transform_input_img(self):
        """Transforms only input image"""
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.2),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
                ToTensorV2()
            ]
        )

    @property
    def _transform_only_target(self):
        """Transforms only target image"""
        return A.Compose(
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
                ToTensorV2()
            ]
        )
