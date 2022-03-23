#!E:\anaconda/python

import os
import csv
import numpy as np


def build_dataset(dir='', output_file=''):
    if dir == '' or output_file == '':
        print("Need to specify the directory that contains images and masks and the output csv file!")
        return
    if not os.path.exists(dir):
        print(f"{dir} does not exist!")
        return

    image_dir = os.path.join(dir, 'images')
    mask_dir = os.path.join(dir, 'annotations_binary')
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print("Image directory or mask directory does not exist!")
        return

    # re-write the csv file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for image_name, mask_name in zip(os.listdir(image_dir), os.listdir(mask_dir)):
            writer.writerow([os.path.join(image_dir, image_name), os.path.join(mask_dir, mask_name)])


def train_test_split(csv_file, test_ratio, seed=42):
    output_dir = os.path.dirname(csv_file)
    train_file = os.path.join(output_dir, 'train.csv')
    test_file = os.path.join(output_dir, 'test.csv')

    # read image pairs from csv file

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

    with open(train_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in train_image_mask_list:
            writer.writerow(line)
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in test_image_mask_list:
            writer.writerow(line)


def get_dataset_list(filename=''):
    if filename == '':
        print("Need to specify which csv file to read!")
        return []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)


if __name__ == '__main__':
    dataset_dir = os.path.join(os.path.dirname(__file__), '../../WildcatCreek-Data')
    output_dir = os.path.join(os.path.dirname(__file__), '../dataset/WildcatCreek-Data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'dataset.csv')

    # write image and mask path pairs as rows into csv file
    # build_dataset(dataset_dir, output_file)

    # write split training and test file names into separate csv files
    # train_test_split(output_file, test_ratio=0.2)

    # test output of dataset read from csv file
    get_dataset_list(output_file)