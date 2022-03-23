#!E:\anaconda/python

from torch.utils.data import Dataset
from torchvision.io import read_image
from src.utils.build_dataset import get_dataset_list
import os


class FluvialDataset(Dataset):
    def __init__(self, dataset_dir, train=True, transform=None, target_transform=None):
        """
        Custom dataset class
        :param dataset_dir: directory of target dataset in src/dataset/
        :param train: whether get the training dataset or test dataset
        :param transform: original image transform
        :param target_transform: mask image transform
        """
        super(FluvialDataset, self).__init__()  # super this class to enable access to Dataset Class functions
        dataset_file = os.path.join(dataset_dir, 'train.csv' if train else 'test.csv')
        dataset_list = get_dataset_list(dataset_file)
        self.img_files = [pair[0] for pair in dataset_list]
        self.mask_files = [pair[1] for pair in dataset_list]
        # print(self.img_files)
        # print(self.mask_files)
        self.transform = transform  # define the transform to be performed on the images at import
        self.target_transform = target_transform  # define the transform to be performed on the masks at import

    def __len__(self):
        """This function returns the length of the dataset (number of images present used for training)
        Input:
            None - Class function
        Output
            Returns the number of images in the dataset """
        # return the length of the dataset for when len() is called
        return len(self.img_files)

    def __getitem__(self, idx):
        """This function of the dataset class returns an item from the dataset. I will in fact return an image from
        the dataset and its corresponding label or segmented image.
        Input
            idx - the index of the image to get from a list of all images/masks
        output
            image -  the transformed image to be used in neural network training
            mask - the transformed mask to be used in neural network training"""
        #  Define the location of the mask and image to get
        img_path = self.img_files[idx]  # pull out the image files associated with
        mask_path = self.mask_files[idx]  # pull out the mask file for the selected image (mask and image files in same order)
        # read the image and mask using the torchvision.io read_image method which returns an image as a
        image = read_image(img_path)
        mask = read_image(mask_path)
        # apply transforms
        if self.transform:  # If image transforms have been specified
            image = self.transform(image)  # apply the transform to the image (e.x. add padding)
        if self.target_transform:  # if mask transforms have been specified
            mask = self.target_transform(mask)  # apply the transform to the image (e.x. add padding)
        return image, mask


if __name__ == '__main__':
    dataset_dir = os.path.join(os.path.dirname(__file__), '../dataset/WildcatCreek-Data')
    training_dataset = FluvialDataset(dataset_dir, train=True)
    test_dataset = FluvialDataset(dataset_dir, train=False)
    print(len(training_dataset))
    print(len(test_dataset))
    # print(training_dataset[0])
    # print(test_dataset[0])