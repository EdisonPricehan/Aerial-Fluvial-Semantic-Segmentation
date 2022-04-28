#!E:\anaconda/python

import matplotlib.pyplot as plt
import os


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    from src.networks.dataset import FluvialDataset

    # define multiple dataset directories to visualize
    dataset_dir1 = os.path.join(os.path.dirname(__file__), '../dataset/WildcatCreek-Data')
    dataset_dir2 = os.path.join(os.path.dirname(__file__), '../dataset/River-Segmentation-Data')

    training_dataset = FluvialDataset(dataset_dir2, train=True)
    print(f"Train dataset size: {len(training_dataset)}")
    img, mask = training_dataset[0]  # plot the first training data pair
    img = img.permute(1, 2, 0)  # change from c x h x w to h x w x c for display
    mask = mask.squeeze()  # reduce from 3d to 2d
    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")
    plot_img_and_mask(img, mask)
