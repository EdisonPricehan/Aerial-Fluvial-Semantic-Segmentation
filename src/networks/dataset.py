#!E:\anaconda/python

from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T
from src.utils.build_dataset import get_dataset_list
import os


class FluvialDataset(Dataset):
    def __init__(self, dataset_dir, train=True, use_augment=True, transform=None, target_transform=None):
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

        # read augmented training set
        augmented_set = []
        if use_augment:
            assert train, "Should not include augmented training data for test data!"
            aug_train_path = os.path.join(dataset_dir, 'train_aug.csv')
            if not os.path.exists(aug_train_path):
                print("Augmented training set is not available, use plain training set!")
            else:
                augmented_set = get_dataset_list(aug_train_path)

        # merge plain and augmented training sets
        dataset_list.extend(augmented_set)

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
        """This function of the dataset class returns an item from the dataset. It will in fact return an image from
        the dataset and its corresponding label or segmented image.
        Input
            idx - the index of the image to get from a list of all images/masks
        output
            image -  the transformed image to be used in neural network training
            mask - the transformed mask to be used in neural network training"""
        #  Define the location of the mask and image to get
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]

        # read the image and mask using the torchvision.io read_image method which returns an image as a tensor
        image = read_image(img_path)
        mask = read_image(mask_path)

        # apply transforms
        if self.transform:
            image = self.transform(image)  # apply the transform to the image (e.x. add padding)
            image = image / 255
            # print(f"image shape {image.shape}")
        if self.target_transform:
            mask = self.target_transform(mask)  # apply the transform to the image (e.x. add padding)

            # convert 3-channel  gray mask to 1-channel mask
            if mask.shape[0] == 3:
                mask = T.Grayscale(num_output_channels=1)(mask)

            mask = mask.squeeze()
            mask = mask / 255

            # print(f"mask shape {mask.shape}")
        return image, mask


if __name__ == '__main__':
    dataset_dir = os.path.join(os.path.dirname(__file__), '../dataset/WildcatCreek-Data')

    from src.utils.custom_transforms import resize
    training_dataset = FluvialDataset(dataset_dir, train=True, use_augment=True,
                                      transform=resize, target_transform=resize)
    test_dataset = FluvialDataset(dataset_dir, train=False, use_augment=False,
                                  transform=resize, target_transform=resize)
    print(len(training_dataset))
    print(len(test_dataset))
    # get the shape of image and mask of first pair
    print(training_dataset[0][0].shape)
    print(training_dataset[0][1].shape)

    # print(training_dataset[1000][0].shape)
    # print(training_dataset[1000][1].shape)

    # print(training_dataset[0])
    # print(test_dataset[0])
