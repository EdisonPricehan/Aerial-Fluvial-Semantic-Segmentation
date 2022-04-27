#!E:\anaconda/python

import os
import shutil
import cv2
import numpy as np


water_rgb = [128, 64, 128]


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


def copy_paste_rename(image_list, target_dir='', prefix=''):
    if target_dir == '':
        print("Need to specify target directory!")
        return
    if os.path.exists(target_dir):
        print(f"{target_dir} already exists.")
    else:
        os.makedirs(target_dir)
        print(f"{target_dir} was created.")

    for i, image in enumerate(image_list):
        _, file_extension = os.path.splitext(image)
        file_name = prefix + '-' + ('%04d' % i) + file_extension
        new_name = os.path.join(target_dir, file_name)
        # print(image)
        # print(new_name)
        try:
            shutil.copy(image, new_name)
        except shutil.SameFileError:
            print(f"{new_name} already exists when paste, continue.")


def binarize_images(image_list, target_dir=''):
    if target_dir == '':
        print("Need to specify target directory!")
        return
    if len(image_list) == 0:
        print("Empty image list!")
        return

    if os.path.exists(target_dir):
        print(f"{target_dir} already exists.")
    else:
        os.makedirs(target_dir)
        print(f"{target_dir} was created.")

    for image_path in image_list:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # water is 255, non-water is 0
        water_mask = ((image[:, :, 0] == water_rgb[0]) & (image[:, :, 1] == water_rgb[1]) & (image[:, :, 2] == water_rgb[2])).astype(np.uint8)
        water_mask *= 255

        # print(water_mask.shape)
        # print(water_mask)
        # cv2.imshow("mask", water_mask)
        # cv2.waitKey(0)

        target_name = os.path.join(target_dir, os.path.basename(image_path))
        print(target_name)
        cv2.imwrite(target_name, water_mask)


def scale_masks_to_visualize(mask_dir, output_dir):
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
        target_mask = (ori_mask * 255).astype(np.uint8)
        target_name = os.path.join(output_dir, os.path.basename(mask_path))
        print(target_name)
        cv2.imwrite(target_name, target_mask)


if __name__ == '__main__':
    root = os.path.dirname(__file__)
    wildcat_creek_dir = os.path.join(root, '../../WildcatCreek-Data')
    print(wildcat_creek_dir)

    # get all images list recursively
    img_list = []
    get_filelist(wildcat_creek_dir, img_list)
    print(len(img_list))

    # get training images list
    data = filter_by_suffix(img_list, '.jpg')
    print(len(data))
    # print(images[:10])

    # get all masks list
    masks = filter_by_name(img_list, name='color_mask')
    print(len(masks))
    # print(masks[:10])

    # paste all wildcat creek images to images directory
    # data_dir = os.path.join(wildcat_creek_dir, 'images')
    # copy_paste_rename(data, target_dir=data_dir, prefix='wildcat')

    # paste all wildcat creek masks to annotations directory
    # masks_dir = os.path.join(wildcat_creek_dir, 'annotations')
    # print(masks_dir)
    # copy_paste_rename(masks, target_dir=masks_dir, prefix='wildcat-mask')

    # binarize masks to water and non-water pixels
    mask_list_path = os.path.join(wildcat_creek_dir, 'annotations')
    print(mask_list_path)
    mask_list = []
    get_filelist(mask_list_path, mask_list)
    print(len(mask_list))
    output_path = os.path.join(wildcat_creek_dir, 'annotations_binary')
    binarize_images(mask_list, target_dir=output_path)

    # convert mask images that range 0-1 to 0-255 to be visualized
    # river_seg_dir = os.path.join(root, '../../River-Segmentation-Data')
    # river_seg_mask_dir = os.path.join(river_seg_dir, 'annotations')
    # river_seg_mask_output_dir = os.path.join(river_seg_dir, 'annotations_binary')
    # scale_masks_to_visualize(river_seg_mask_dir, river_seg_mask_output_dir)

