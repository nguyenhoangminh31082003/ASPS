import os

import albumentations as albu
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def get_train_augmentation(img_size, ver):
    if ver == 1:
        transforms = albu.Compose(
            [
                albu.Resize(img_size, img_size, always_apply=True),
                albu.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    if ver == 2:
        transforms = albu.Compose(
            [
                albu.OneOf(
                    [albu.HorizontalFlip(), albu.VerticalFlip(), albu.RandomRotate90()],
                    p=0.5,
                ),
                albu.OneOf(
                    [
                        albu.RandomBrightnessContrast(),
                        albu.RandomGamma(),
                        albu.RandomBrightness(),
                    ],
                    p=0.5,
                ),
                albu.OneOf(
                    [
                        albu.MotionBlur(blur_limit=5),
                        albu.MedianBlur(blur_limit=5),
                        albu.GaussianBlur(blur_limit=5),
                        albu.GaussNoise(var_limit=(5.0, 20.0)),
                    ],
                    p=0.5,
                ),
                albu.Resize(img_size, img_size, always_apply=True),
                albu.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    return transforms


def get_test_augmentation(img_size):
    return albu.Compose(
        [
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


class PraNetDataset(Dataset):
    def __init__(self, root_dir, image_size, train, train_aug_ver=None):
        self.gts, self.images = [], []
        self.img_size = image_size

        self.load_dataset(root_dir, train)

        if train:
            self.transforms = get_train_augmentation(image_size, train_aug_ver)
        else:
            self.transforms = get_test_augmentation(image_size)

    def load_dataset(self, root_dir, train):
        img_path = os.path.join(root_dir, "image")
        mask_path = os.path.join(root_dir, "masks")

        img_list, mask_list = [], []

        for image in os.listdir(img_path):
            img_list.append(os.path.join(img_path, image))
            mask_list.append(os.path.join(mask_path, image))

        for image_name, gt_name in zip(img_list, mask_list):
            self.images.append(image_name)
            self.gts.append(gt_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.gts[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transforms:
            transformed = self.transforms(image=image, masks=[mask])
            image = transformed["image"]
            mask = transformed["masks"][0]

            mask = mask / 255.0
            mask = torch.unsqueeze(mask, 0)
            mask = mask.type_as(image)

        return image, mask, self.images[index]
