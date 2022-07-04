#!E:\anaconda/python

import os
import csv
import numpy as np
import fire
import cv2
from tqdm import tqdm


def train_valid_test_split(csv_file, test_ratio, valid_ratio=0, seed=42):
    """
    split train, validation and test filepaths as separate csv files
    :param csv_file: csv file of all (image, mask) pairs to be split
    :param test_ratio: ratio of test set
    :param valid_ratio: ratio of validation set
    :param seed: some fixed integer to allow repeatability
    :return:
    """
    # check args validity
    if test_ratio + valid_ratio > 1.0 or test_ratio < 0.0 or valid_ratio < 0.0:
        print("Invalid test/validation ratio!")
        return

    # get absolute path and target directory
    csv_file_abs = os.path.join(os.path.dirname(__file__), csv_file)
    output_dir = os.path.dirname(csv_file_abs)

    # train and test csv files will be stored in the same directory with the dataset csv file
    train_file = os.path.join(output_dir, 'train.csv')
    valid_file = os.path.join(output_dir, 'valid.csv')
    test_file = os.path.join(output_dir, 'test.csv')

    # read image pairs from csv file, get subset as list
    with open(csv_file_abs, 'r') as f:
        reader = csv.reader(f)
        reader_list = list(reader)
        line_num = len(reader_list)
        print(f"Splitting {line_num} data into train, valid and test sets ...")
        np.random.seed(seed)
        remaining_indices = range(line_num)
        test_indices = np.random.choice(remaining_indices, np.floor(test_ratio * line_num).astype(int), replace=False)
        remaining_indices = list(set(remaining_indices).difference(set(test_indices)))
        valid_indices = np.random.choice(remaining_indices, np.floor(valid_ratio * line_num).astype(int), replace=False)
        train_indices = list(set(remaining_indices).difference(set(valid_indices)))

        # validity check
        assert len(test_indices) + len(valid_indices) + len(train_indices) == line_num, \
            f"Overflow after splitting! test {len(test_indices)}, valid {len(valid_indices)}, train {len(train_indices)}"
        assert set(train_indices).isdisjoint(set(test_indices)), \
               f"Test and train indices overlap! {list(set(test_indices).intersection(set(train_indices)))}"
        assert set(test_indices).isdisjoint(set(valid_indices)), \
               f"Test and valid indices overlap! {list(set(test_indices).intersection(set(valid_indices)))}"
        assert set(valid_indices).isdisjoint(set(train_indices)), \
               f"Valid and train indices overlap! {list(set(valid_indices).intersection(set(train_indices)))}"

        # form csv lines as list for each set
        test_image_mask_list = [reader_list[i] for i in test_indices]
        valid_image_mask_list = [reader_list[i] for i in valid_indices]
        train_image_mask_list = [reader_list[i] for i in train_indices]

    # write subset of train, validation and test to its csv file
    with open(train_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in train_image_mask_list:
            writer.writerow(line)
    with open(valid_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in valid_image_mask_list:
            writer.writerow(line)
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in test_image_mask_list:
            writer.writerow(line)


def build_test_set(dataset, trainset, testset):
    """
    Build test set from the given dataset and trainset and store to testset (all relative paths)
    :param dataset: relative path of the whole dataset csv file
    :param trainset: relative path of the trainset csv file
    :param testset: relative path of the testset csv file as output
    """
    dataset_path = os.path.join(os.path.dirname(__file__), dataset)
    trainset_path = os.path.join(os.path.dirname(__file__), trainset)
    testset_path = os.path.join(os.path.dirname(__file__), testset)
    if not os.path.exists(dataset_path) or not os.path.exists(trainset_path):
        print("Dataset or trainset directory does not exist!")
        return

    # get list of image-mask pairs from both dataset and trainset and get the difference set
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        dataset_list = list(reader)
        dataset_list = [tuple(sub_list) for sub_list in dataset_list]  # convert to tuple
        dataset_set = set(dataset_list)
    with open(trainset_path, 'r') as f:
        reader = csv.reader(f)
        trainset_list = list(reader)
        trainset_list = [tuple(sub_list) for sub_list in trainset_list]  # convert to tuple
        trainset_set = set(trainset_list)
    testset_set = dataset_set.difference(trainset_set)

    # write testset to csv file
    with open(testset_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in testset_set:
            writer.writerow(line)

    print("Testset csv built!")


def get_dataset_list(filename=''):
    """
    get list of (image, mask) tuple from csv file
    :param filename: absolute path of csv file
    :return: [(image, mask)]
    """
    if filename == '':
        print("Need to specify which csv file to read!")
        return []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)


def build_csv_from_datasets(dataset_dir_list,
                            image_dir='images',
                            mask_dir='annotations_binary',
                            output_dirname='dataset_csv',
                            output_filename='dataset.csv'):
    """
    Choose one or many dataset directories and form csv file to store the image and mask paths.
    Easy to modify the total datasets for different experiments that may be based on various combinations of multiple
    datasets, and easy for different train-test split ratios since they all doing things on csv file without any
    copy or move of original datasets
    :param dataset_dir_list: list of relative paths to all dataset directories that are interested
    :param image_dir: inside each dataset directory, the name of subdirectory that stores original images
    :param mask_dir: inside each dataset directory, the name of subdirectory that stores masks
    :param output_dirname: used to illustrate the purpose or experimental composition of the datasets used
    :param output_filename: name of the target csv file
    :return:
    """
    dataset_num = len(dataset_dir_list)
    if dataset_num == 0:
        print("Need a non-empty list of all dataset directories' relative paths!")
        return
    print(f"Start building csv file from {dataset_num} datasets ...")

    # define output csv file directory
    output_dir = os.path.join(os.path.dirname(__file__), '../dataset', output_dirname)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # define output csv file path
    output_filepath = os.path.join(output_dir, output_filename)
    if os.path.exists(output_filepath):
        print(f"{output_filepath} already exists, please delete it and try again!")
        return

    # loop over each dataset directory and append image-mask pairs to the target csv file
    for dataset_name in dataset_dir_list:
        # define dataset path
        dataset_dir = os.path.join(os.path.dirname(__file__), dataset_name)
        assert os.path.exists(dataset_dir), "Dataset directory does not exist!"
        print(f"Building for {dataset_dir} ...")

        # make sure both images and masks directories exist
        image_dir_abs = os.path.join(dataset_dir, image_dir)
        mask_dir_abs = os.path.join(dataset_dir, mask_dir)
        if not os.path.exists(image_dir_abs) or not os.path.exists(mask_dir_abs):
            print("Image directory or mask directory does not exist!")
            return

        # append to the csv file
        with open(output_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            for image_name, mask_name in zip(os.listdir(image_dir_abs), os.listdir(mask_dir_abs)):
                # print(f"{image_name=} {mask_name=}")
                writer.writerow([os.path.join(image_dir_abs, image_name), os.path.join(mask_dir_abs, mask_name)])

    print("Csv file building finished!")


def build_statistics_from_datasets(dataset_list, output_path):
    """
    Build statistics of the given dataset directories
    :param output_path: relative path of the target csv file
    :param dataset_list: list of relative paths to all dataset csv files that are interested
    :return:
    """
    from constants import water_rgb_aerial, vegetation_rgb, dry_sediment_rgb, sky_rgb, self_rgb, wood_in_river_rgb, \
        boat_rgb, bridge_rgb

    # Validity check
    dataset_num = len(dataset_list)
    if dataset_num == 0:
        print("Need a non-empty list of all dataset directories' relative paths!")
        return
    print(f"Start building statistics from {dataset_num} datasets ...")

    # construct absolute csv file output path
    output_path_abs = os.path.join(os.path.dirname(__file__), output_path)
    print(f"Output path: {output_path_abs}")

    # write header to the csv file
    with open(output_path_abs, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['water', 'vegetation', 'sediment', 'sky', 'self', 'obstacles', 'boat', 'bridge'])

    # construct rgb list for all classes with the same order as the header
    rgb_list = [water_rgb_aerial, vegetation_rgb, dry_sediment_rgb, sky_rgb,
                self_rgb, wood_in_river_rgb, boat_rgb, bridge_rgb]

    # loop over each dataset directory and write pixel numbers of each class to the target csv file
    for dataset_path in dataset_list:
        # define dataset path and check its existence
        dataset_path_abs = os.path.join(os.path.dirname(__file__), dataset_path)
        assert os.path.exists(dataset_path_abs), "Dataset directory does not exist!"

        # read image-mask pairs from the csv file
        image_mask_list = get_dataset_list(dataset_path_abs)
        print(f"Building statistics for {len(image_mask_list)} masks from {dataset_path_abs} ...")

        with open(output_path_abs, 'a',  newline='') as f:
            writer = csv.writer(f)
            for image_mask_path in tqdm(image_mask_list):
                mask = cv2.imread(image_mask_path[1])  # read mask
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # convert to rgb
                h, w, _ = mask.shape
                print(f"RGB Mask size: {h}x{w}")
                class_count_list = []
                for rgb in rgb_list:
                    mask_binary = ((mask[:, :, 0] == rgb[0]) &
                                   (mask[:, :, 1] == rgb[1]) &
                                   (mask[:, :, 2] == rgb[2])).astype(np.uint8)
                    count = np.sum(mask_binary)  # sum all pixels of 2d array
                    class_count_list.append(count)
                assert len(class_count_list) == len(rgb_list)
                writer.writerow(class_count_list)
                assert sum(class_count_list) == h * w, \
                    f"Pixel number mismatch, all classes {class_count_list} {sum(class_count_list)}, all image {h * w}!"

    print("Statistics csv file building finished!")


def check_dataset_validity(dataset_csv_path):
    """
    Check the validity of the dataset before training
    Make sure all image-mask pairs exist, and all images are rgb
    :param dataset_csv_path: relative path to the dataset csv file
    :return:
    """
    from torchvision.io import read_image

    # get absolute path
    dataset_csv_path_abs = os.path.join(os.path.dirname(__file__), dataset_csv_path)
    assert os.path.exists(dataset_csv_path_abs), "Dataset csv file does not exist!"

    # read image-mask pairs from the csv file
    image_mask_list = get_dataset_list(dataset_csv_path)
    print(f"Checking validity of {len(image_mask_list)} image-mask pairs from {dataset_csv_path_abs} ...")

    # loop over each image-mask pair and check several criteria
    for image_mask_path in tqdm(image_mask_list):
        assert os.path.exists(image_mask_path[0]), "Image does not exist!"
        assert os.path.exists(image_mask_path[1]), "Mask does not exist!"
        image = read_image(image_mask_path[0])
        mask = read_image(image_mask_path[1])
        assert image.shape[0] == 3, f"{image_mask_path[0]} is not rgb!"
        assert image.shape[1:] == mask.shape[-2:], f"{image_mask_path[0]} and its mask are not the same size!"

    print(f"{dataset_csv_path_abs} dataset validity check passed!")


if __name__ == '__main__':
    fire.Fire()

    #### example usage 1 of build_csv_from_datasets ####
    # python build_dataset.py
    # build_csv_from_datasets
    # "['../../Deep-Learning-Data/Dartmouth_dataset',
    # '../../Deep-Learning-Data/diVeny_dataset',
    # '../../Deep-Learning-Data/Eamont_dataset',
    # '../../Deep-Learning-Data/Kananaskis_dataset',
    # '../../Deep-Learning-Data/Kingie_dataset',
    # '../../Deep-Learning-Data/Kinogawa_dataset',
    # '../../Deep-Learning-Data/Kurobe_dataset',
    # '../../Deep-Learning-Data/Pacuare_dataset',
    # '../../Deep-Learning-Data/Quelle_dataset',
    # '../../Deep-Learning-Data/Sesia_dataset',
    # '../../Deep-Learning-Data/StMarg_dataset']"
    # 'images'
    # 'annotations_tif'
    # '11-rivers-original'
    # 'train.csv'
    #### example usage 1 of build_csv_from_datasets ####

    #### example usage 2 of build_csv_from_datasets ####
    # python build_dataset.py
    # build_csv_from_datasets
    # "['../../WildcatCreek-Data/wildcat_dataset']"
    # 'images'
    # 'annotations_binary'
    # 'WildcatCreek-Data'
    # 'dataset.csv'
    #### example usage 2 of build_csv_from_datasets ####

    #### example usage 3 of build_statistics_from_datasets ####
    # python build_dataset.py
    # build_statistics_from_datasets
    # "['../dataset/WabashRiver-Data/dataset_rgb.csv',
    # '../dataset/WildcatCreek-Data/dataset_rgb.csv',
    # '../dataset/Bridges/dataset_4bridges_rgb.csv']"
    # '../dataset/Wabash-Wildcat/statistics.csv'
    #### example usage 3 of build_statistics_from_datasets ####

    #### example usage 4 of check_dataset_validity ####
    # python build_dataset.py
    # check_dataset_validity
    # '../dataset/3-datasets-baseline/train.csv'
    #### example usage 4 of check_dataset_validity ####

    #### example usage 5 of train_valid_test_split ####
    # python build_dataset.py
    # train_valid_test_split
    # '../dataset/WildcatCreek-Data/dataset.csv'
    # 0.4
    # 0.2
    #### example usage 5 of train_valid_test_split ####
