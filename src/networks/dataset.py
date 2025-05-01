#!E:\anaconda/python

import os
import numpy as np
import torch
import pandas as pd
from typing import List, Tuple, Callable

from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T

from utils.build_dataset import get_dataset_list
from utils.convert_images import abs_path


__all__ = ['FluvialDataset', 'VideoDataset', 'VideoLabelDataset']


class FluvialDataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            use_augment: bool = False,
            multi_class: bool = False,
            transform=None,
            target_transform=None,
    ):
        """
        Custom dataset class.

        Args:
            dataset_path: Relative path of target dataset csv file in src/dataset/
            use_augment: Whether augment the dataset.
            multi_class: Whether the mask is multi class or binary.
            transform: Transform to the image.
            target_transform: Transform to the mask.
        """
        super(FluvialDataset, self).__init__()

        dataset_file = abs_path(dataset_path)
        assert os.path.exists(dataset_file), f'{dataset_file} does not exist!'

        dataset_list = get_dataset_list(dataset_file)

        # read augmented training set
        augmented_set = []
        # if use_augment:
        #     assert train, "Should not include augmented training data for test data!"
        #     aug_train_path = os.path.join(dataset_dir, 'train_aug.csv')
        #     if not os.path.exists(aug_train_path):
        #         print("Augmented training set is not available, use plain training set!")
        #     else:
        #         augmented_set = get_dataset_list(aug_train_path)

        # merge plain and augmented training sets
        dataset_list.extend(augmented_set)

        self.img_files = [pair[0] for pair in dataset_list]
        self.mask_files = [pair[1] for pair in dataset_list]
        # print(self.img_files)
        # print(self.mask_files)
        self.transform = transform
        self.target_transform = target_transform
        self.multi_class = multi_class

    def __len__(self) -> int:
        """
        This function returns the length of the dataset (number of images present used for training)
        Input:
            None - Class function
        Output
            Returns the number of images in the dataset
        """
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """
        This function of the dataset class returns an item from the dataset. It will in fact return an image from
        the dataset and its corresponding label or segmented image.
        Input
            idx - the index of the image to get from a list of all images/masks
        output
            image -  the transformed image to be used in neural network training
            mask - the transformed mask to be used in neural network training
        """
        # Define the location of the mask and image to get
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]

        # The read image and mask are tensors of shape (C, H, W)
        image = read_image(img_path)
        mask = read_image(mask_path)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            image = image / 255
            # print(f"image shape {image.shape}")
        if self.target_transform:
            mask = self.target_transform(mask)

            # convert 3-channel gray mask to 1-channel mask
            if mask.shape[0] == 3 and not self.multi_class:
                mask = T.Grayscale(num_output_channels=1)(mask)

            mask = mask.squeeze()  # remove redundant dimension
            mask = mask / 255

            # print(f"mask shape {mask.shape}")
        return image, mask


class VideoDataset(Dataset):
    """ Video Dataset for loading video.
        It will output only path of video (neither video file path or video folder path).
        However, you can load video as torch.Tensor (C x L x H x W).
        See below for an example of how to read video as torch.Tensor.
        Your video dataset can be image frames or video files.

    Args:
        csv_file (str): path fo csv file which store path of video file or video folder.
            the format of csv_file should like:

            # example_video_file.csv   (if the videos of dataset is saved as video file)

            path
            ~/path/to/video/file1.mp4
            ~/path/to/video/file2.mp4
            ~/path/to/video/file3.mp4
            ~/path/to/video/file4.mp4

            # example_video_folder.csv   (if the videos of dataset is saved as image frames)

            path
            ~/path/to/video/folder1/
            ~/path/to/video/folder2/
            ~/path/to/video/folder3/
            ~/path/to/video/folder4/

    Example:

        if the videos of dataset is saved as video file

        if the video of dataset is saved as frames in video folder
        The tree like: (The names of the images are arranged in ascending order of frames)
        ~/path/to/video/folder1
        ├── frame-001.jpg
        ├── frame-002.jpg
        ├── frame-003.jpg
        └── frame-004.jpg
    """

    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """ get a video """
        video = self.dataframe.iloc[index].path
        if self.transform:
            video = self.transform(video)
        return video


class VideoLabelDataset(Dataset):
    """ Dataset Class for Loading Video.
        It will output path and label. However, you can load video as torch.Tensor (C x L x H x W).
        See below for an example of how to read video as torch.Tensor.
        You can load tensor from video file or video folder by using the same way as VideoDataset.

    Args:
        csv_file (str): path fo csv file which store path and label of video file (or video folder).
            the format of csv_file should like:

            path, label
            ~/path/to/video/file1.mp4, 0
            ~/path/to/video/file2.mp4, 1
            ~/path/to/video/file3.mp4, 0
            ~/path/to/video/file4.mp4, 2
    """

    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """ get a video and its label """
        video = self.dataframe.iloc[index].path
        label = self.dataframe.iloc[index].label
        if self.transform:
            video = self.transform(video)
        return video, label


if __name__ == '__main__':
    # test for FluvialDataset
    train_dataset_path = '../dataset/3-datasets-baseline/train.csv'
    valid_dataset_path = '../dataset/3-datasets-baseline/valid_wabash_wildcat.csv'
    from utils.custom_transforms import resize
    training_dataset = FluvialDataset(train_dataset_path, use_augment=False, transform=resize, target_transform=resize)
    valid_dataset = FluvialDataset(valid_dataset_path, use_augment=False, transform=resize, target_transform=resize)
    print(len(training_dataset))
    print(len(valid_dataset))
    # get the shape of image and mask of first pair
    print(training_dataset[0][0].shape)
    print(training_dataset[0][1].shape)

    # test for VideoDataset
    from utils import video_transforms as transforms
    import torchvision
    dataset = VideoDataset('./video-path/video_path.csv',
                           transform=torchvision.transforms.Compose([
                            transforms.VideoFilePathToTensor(max_len=50, fps=1),
                            transforms.VideoResize([320, 544])]))
    print(f"dataset len: {len(dataset)}")
    video = dataset[0]
    print(f"video size: {video.size()}")
    from torch.utils.data import DataLoader
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for video in test_loader:
        print(f"{video=}")
        break

    # test for VideoLabelDataset
    # dataset = VideoLabelDataset(
    #     './video-path/video_path.csv',
    #     transform=torchvision.transforms.Compose([
    #         transforms.VideoFilePathToTensor(max_len=10, fps=10, padding_mode='last'),
    #         transforms.VideoResize([320, 544]),
    #     ])
    # )
    # video, label = dataset[0]
    # print(video.size(), label)
    # frame1 = torchvision.transforms.ToPILImage()(video[:, 29, :, :])
    # frame2 = torchvision.transforms.ToPILImage()(video[:, 39, :, :])
    # frame1.show()
    # frame2.show()
    #
    # test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for videos, labels in test_loader:
    #     print(videos.size(), label)
