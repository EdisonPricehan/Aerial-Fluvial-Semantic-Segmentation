#!E:\anaconda/python

import os
import shutil
import cv2
import numpy as np
import fire
from skimage import io
from tqdm import tqdm

from constants import *


def get_filelist(dir, Filelist):
    """
    Recursively get all files under dir
    :param dir:
    :param Filelist:
    :return:
    """
    if os.path.isfile(dir):
        Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist


def filter_by_suffix(image_list, suffix=''):
    new_list = []
    if suffix == '':
        print("Cannot filter empty suffix.")
        return new_list

    for i in image_list:
        filename, file_extension = os.path.splitext(i)
        if file_extension == suffix:
            new_list.append(i)
    return new_list


def filter_by_name(image_list, name=''):
    new_list = []
    if name == '':
        print("Cannot filter empty name, return original list.")
        return image_list

    for i in image_list:
        if name in i:
            new_list.append(i)
    return new_list


def abs_path(rela_path):
    absolute_path = os.path.join(os.path.dirname(__file__), rela_path)
    return absolute_path


def copy_paste_rename(input_dir, target_dir, keep_every_n=1, rename=False, prefix=''):
    """
    Copy and paste files from input_dir to target_dir at every n image, and rename them sequentially if wanted.
    :param input_dir: relative path to input directory
    :param target_dir: relative path to target directory
    :param rename: whether to rename files
    :param keep_every_n: keep every n image
    :param prefix: prefix to add to the file name if rename is True
    :return:
    """
    # get absolute paths for input and target directories
    input_dir_abs = abs_path(input_dir)
    target_dir_abs = abs_path(target_dir)

    # check if input directory exists
    if not os.path.exists(input_dir_abs):
        print(f"{input_dir_abs} does not exist!")
        return

    # check if target directory exists, make it if not
    if os.path.exists(target_dir_abs):
        print(f"{target_dir_abs} already exists.")
    else:
        os.makedirs(target_dir_abs)
        print(f"{target_dir_abs} was created.")

    # get all files in input directory
    image_list = []
    get_filelist(input_dir_abs, image_list)

    # copy, paste and rename images at every n image
    for i, image_path in tqdm(enumerate(image_list)):
        if i % keep_every_n == 0:
            if rename:
                _, file_extension = os.path.splitext(image_path)
                file_name = prefix + '-' + ('%04d' % i) + file_extension
                new_path = os.path.join(target_dir_abs, file_name)
            else:
                new_path = os.path.join(target_dir_abs, os.path.basename(image_path))

            # copy and paste image
            try:
                shutil.copy(image_path, new_path)
            except shutil.SameFileError:
                print(f"{new_path} already exists when paste, continue.")

    print(f"Copy, paste and rename finished for {len(image_list)} images.")


def binarize_masks(mask_dir, target_dir='', water_rgb=None):
    """
    Binarize multi-class RGB masks to binary masks (0 as non-water and 255 as water)
    :param mask_dir: relative path to rgb masks directory
    :param target_dir: relative path to binary masks directory
    :param water_rgb: rgb list of water pixel
    :return:
    """
    # set default water RGB if not given
    if water_rgb is None:
        water_rgb = water_rgb_aerial

    # set absolute paths for input and output directories
    mask_dir = abs_path(mask_dir)
    if target_dir == '':
        # if target directory not specified, make it the same-level with mask directory and use a fixed directory name
        target_dir = os.path.join(mask_dir, '../annotations_binary')
    else:
        target_dir = abs_path(target_dir)

    # cannot proceed without input mask directory
    if not os.path.exists(mask_dir):
        print(f"{mask_dir} does not exist!")
        return

    # get all masks paths as a list
    mask_list = []
    get_filelist(mask_dir, mask_list)
    print(f"Masks num: {len(mask_list)}")
    if len(mask_list) == 0:
        print("Empty mask list!")
        return

    if os.path.exists(target_dir):
        print(f"{target_dir} already exists.")
    else:
        os.makedirs(target_dir)
        print(f"{target_dir} was created.")

    # binarize masks
    for mask_path in mask_list:
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # make water 255, non-water 0
        water_mask = ((mask[:, :, 0] == water_rgb[0]) &
                      (mask[:, :, 1] == water_rgb[1]) &
                      (mask[:, :, 2] == water_rgb[2])).astype(np.uint8)
        water_mask *= 255

        # save binary mask to the desired target directory
        target_name = os.path.join(target_dir, os.path.basename(mask_path))
        cv2.imwrite(target_name, water_mask)
        print(f"Saved binary mask to {target_name}.")

    print(f"Binarization finished for all {len(mask_list)} masks!")


def scale_masks_to_visualize(mask_dir, output_dir):
    """
    Scale 0-1 pixel values to 0-255 for visualization
    :param mask_dir: relative path to 0-1 masks directory
    :param output_dir: relative path to 0-255 masks directory
    :return:
    """
    # set absolute paths for input and output directories
    mask_dir = abs_path(mask_dir)
    output_dir = abs_path(output_dir)
    assert os.path.exists(mask_dir), "Mask directory does not exist"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{output_dir} was created.")

    ori_mask_list = []
    get_filelist(mask_dir, ori_mask_list)
    if len(ori_mask_list) == 0:
        print("Empty mask list!")
        return

    for mask_path in ori_mask_list:
        ori_mask = cv2.imread(mask_path)
        target_mask = (ori_mask * 255).astype(np.uint8)[:, :, 0]
        print(f"Target mask shape: {target_mask.shape}")
        target_name = os.path.join(output_dir, os.path.basename(mask_path))
        print(f"Target mask path: {target_name}")
        cv2.imwrite(target_name, target_mask)

    print(f"Scaling finished for {len(ori_mask_list)} masks.")


def tiff2png(input_dir, output_dir):
    """
    Convert tiff images to png images
    :param input_dir: relative path to tiff images directory
    :param output_dir: relative path to png images directory
    :return:
    """
    input_dir = abs_path(input_dir)
    output_dir = abs_path(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{output_dir} was created.")

    input_list = []
    get_filelist(input_dir, input_list)

    for i in tqdm(input_list):
        im = io.imread(i, as_gray=True)
        print(f"image shape is {im.shape}")

        # convert image pixels from labels to rgb colors
        label2rgb = {1: water_rgb_aerial,  # water
                     2: dry_sediment_rgb,  # Sediment
                     3: vegetation_rgb,  # Green Veg.
                     4: vegetation_rgb,  # Senesc. Veg.
                     5: bridge_rgb  # Paved Road
                     }

        im = im.astype(np.uint8)
        im_rgb = np.dstack((im, im, im)).astype(np.uint8)
        for label, rgb in label2rgb.items():
            idx = (im == label)
            im_rgb[idx] = rgb
        im_rgb = im_rgb.astype(np.uint8)

        # set output file name
        base_name = os.path.splitext(os.path.basename(i))[0]
        base_name_ext = base_name + '.png'
        o = os.path.join(output_dir, base_name_ext)

        io.imsave(o, im_rgb)
    print("Conversion finished!")


if __name__ == '__main__':
    fire.Fire()

    #### example usage 1 ####
    # python convert_images.py
    # tiff2png
    # '../../WabashRiverAerial-Data/wabash_dataset/test_1_masks'
    # '../../WabashRiverAerial-Data/wabash_dataset/test_png_masks'
    #### example usage 1 ####

    #### example usage 2 ####
    # python convert_images.py
    # binarize_masks
    # '../../Deep-Learning-Data/11_rivers_dataset/annotations'
    #### example usage 2 ####

    #### example usage 3 ####
    # python convert_images.py
    # copy_paste_rename
    # '../../Lantern-Pole-Data/all_dataset/images'
    # '../../Lantern-Pole-Data/curated_dataset/images'
    # 8
    #### example usage 3 ####

