#!E:\anaconda/python

import torchvision.transforms.functional as F
import os


def resize(image_tensor, size=300):
    # print(f"Original shape: {image_tensor.shape}")
    new_image_tensor = F.resize(image_tensor, size=size)
    # print(f"Resized shape: {new_image_tensor.shape}")
    return new_image_tensor


if __name__ == '__main__':
    from src.networks.dataset import FluvialDataset
    dataset_dir = os.path.join(os.path.dirname(__file__), '../dataset/WildcatCreek-Data')
    training_dataset = FluvialDataset(dataset_dir, train=True)
    print(len(training_dataset))
    img, mask = training_dataset[0]  # plot the first training data pair
    small_edge_len = 300
    resized_img = resize(img, small_edge_len)
    resized_mask = resize(mask, small_edge_len)

    # plot the images
    import visualizer
    # visualizer.plot_img_and_mask(img.permute(1, 2, 0), mask.squeeze())
    visualizer.plot_img_and_mask(resized_img.permute(1, 2, 0), resized_mask.squeeze())
