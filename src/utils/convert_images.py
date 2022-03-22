#!E:\anaconda/python

import os
import shutil


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


if __name__ == '__main__':
    root = os.path.dirname(__file__)
    wildcat_creek_dir = os.path.join(root, '../../WildcatCreek-Data')
    print(wildcat_creek_dir)

    # get all images list
    img_list = []
    get_filelist(wildcat_creek_dir, img_list)
    print(len(img_list))

    # get training images list
    data = filter_by_suffix(img_list, '.jpg')
    print(len(data))
    # print(images[:10])

    # get all masks list
    masks = filter_by_name(img_list, name='watershed_mask')
    print(len(masks))
    # print(masks[:10])

    # paste all wildcat creek images to images directory
    data_dir = os.path.join(wildcat_creek_dir, 'images')
    copy_paste_rename(data, target_dir=data_dir, prefix='wildcat')

    # paste all wildcat creek masks to annotations directory
    masks_dir = os.path.join(wildcat_creek_dir, 'annotations')
    copy_paste_rename(masks, target_dir=masks_dir, prefix='wildcat-mask')


