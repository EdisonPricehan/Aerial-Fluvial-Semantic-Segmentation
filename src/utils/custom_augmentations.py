#!E:\anaconda/python

import os
import random
import albumentations as A
import cv2
from src.utils.build_dataset import get_dataset_list
from visualizer import *


# define augmentation transforms
aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.2),
    # A.OneOf([
    #     A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
    #     # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
    #     A.Perspective(p=0.5)
    #     ], p=0.8),
    # A.CLAHE(p=0.8),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.5)])


# check existence of the absolute directory path, make directory if it does not exist
def check_dir_and_make(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return True


if __name__ == '__main__':
    # init input data directory and output data directory
    dataset_dir_relative = '../dataset/WildcatCreek-Data'
    trainset_path = os.path.join(os.path.dirname(__file__), dataset_dir_relative, 'train.csv')
    output_dir = '../../WildcatCreek-Data'
    aug_img_output_dir = os.path.join(os.path.dirname(__file__), output_dir, 'images_aug')
    aug_mask_output_dir = os.path.join(os.path.dirname(__file__), output_dir, 'annotations_binary_aug')

    # make data output directories
    check_dir_and_make(aug_img_output_dir)
    check_dir_and_make(aug_mask_output_dir)

    # create lists for all training images and masks
    dataset_list = get_dataset_list(trainset_path)
    img_files = [pair[0] for pair in dataset_list]
    mask_files = [pair[1] for pair in dataset_list]

    # loop over all image-mask pairs and do augmentations
    assert len(img_files) == len(mask_files), "Images and masks not equal in number!"
    loop = 10
    print(f"Start augmenting for {loop} rounds ...")
    for img_path, mask_path in zip(img_files, mask_files):
        image, mask = cv2.imread(img_path), cv2.imread(mask_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # augment each image-mask pair for *loop* times
        for i in range(loop):
            # for comparison
            # image_aug, mask_aug = augment_and_show(aug, image, mask, filename=aug_img_output_dir + '/aug_medium.jpg')

            augmented = aug(image=image, mask=mask)
            image_aug, mask_aug = augmented['image'], augmented['mask']
            aug_img_name = os.path.splitext(os.path.basename(img_path))[0] + '-' + str(i)
            aug_mask_name = os.path.splitext(os.path.basename(mask_path))[0] + '-' + str(i)

            # construct absolute path for augmented image and mask
            aug_img_path = os.path.join(aug_img_output_dir,
                                        aug_img_name + os.path.splitext(os.path.basename(img_path))[1])
            aug_mask_path = os.path.join(aug_mask_output_dir,
                                         aug_mask_name + os.path.splitext(os.path.basename(mask_path))[1])

            # save augmented image and mask
            cv2.imwrite(aug_img_path, image_aug)
            cv2.imwrite(aug_mask_path, mask_aug)

    print(f"Augmented {len(img_files) * loop} image-mask pairs, finished!")
