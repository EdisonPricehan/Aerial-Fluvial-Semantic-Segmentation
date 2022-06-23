#!E:\anaconda/python

import os
import csv
import numpy as np
import fire


def train_test_split(csv_file, test_ratio, seed=42):
    """
    split train and test filepaths as separate csv files
    :param csv_file: csv file of all (image, mask) pairs to be split
    :param test_ratio: ratio of test set
    :param seed: some fixed integer to allow repeatability
    :return:
    """
    output_dir = os.path.dirname(csv_file)

    # train and test csv files will be stored in the same directory with the dataset csv file
    train_file = os.path.join(output_dir, 'train.csv')
    test_file = os.path.join(output_dir, 'test.csv')

    # read image pairs from csv file, get subset as list
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        reader_list = list(reader)
        line_num = len(reader_list)
        print(line_num)
        np.random.seed(seed)
        test_indices = np.random.choice(range(line_num), np.floor(test_ratio * line_num).astype(int))
        train_indices = list(set(range(line_num)).difference(set(test_indices)))
        # print(test_indices)
        test_image_mask_list = [reader_list[i] for i in test_indices]
        train_image_mask_list = [reader_list[i] for i in train_indices]

    # write subset of train and test to its csv file
    with open(train_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in train_image_mask_list:
            writer.writerow(line)
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in test_image_mask_list:
            writer.writerow(line)


def build_test_set(dataset, trainset, testset):
    """
    Build test set from the given dataset and trainset and store to testset (all relative paths)
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


if __name__ == '__main__':
    fire.Fire()

    #### example usage 1 ####

    # python build_dataset.py build_csv_from_datasets
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

    #### example usage 1 ####