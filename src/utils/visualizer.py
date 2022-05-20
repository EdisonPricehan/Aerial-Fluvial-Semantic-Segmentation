#!E:\anaconda/python

import matplotlib.pyplot as plt
from skimage.color import label2rgb
import os
import cv2


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


BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2, **kwargs):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img


def visualize_titles(img, bbox, title, color=BOX_COLOR, thickness=2, font_thickness = 2, font_scale=0.35, **kwargs):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    ((text_width, text_height), _) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, title, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR,
                font_thickness, lineType=cv2.LINE_AA)
    return img


def augment_and_show(aug, image, mask=None, bboxes=None, categories=None, category_id_to_name=None, filename=None,
                     font_scale_orig=0.35, font_scale_aug=0.35, show_title=False, **kwargs):
    augmented = aug(image=image, mask=mask)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)

    # for bbox in bboxes:
    #     visualize_bbox(image, bbox, **kwargs)
    #
    # for bbox in augmented['bboxes']:
    #     visualize_bbox(image_aug, bbox, **kwargs)
    #
    # if show_title:
    #     for bbox, cat_id in zip(bboxes, categories):
    #         visualize_titles(image, bbox, category_id_to_name[cat_id], font_scale=font_scale_orig, **kwargs)
    #     for bbox, cat_id in zip(augmented['bboxes'], augmented['category_id']):
    #         visualize_titles(image_aug, bbox, category_id_to_name[cat_id], font_scale=font_scale_aug, **kwargs)

    if mask is None:
        f, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].imshow(image)
        ax[0].set_title('Original image')

        ax[1].imshow(image_aug)
        ax[1].set_title('Augmented image')
    else:
        f, ax = plt.subplots(2, 2, figsize=(16, 16))

        if len(mask.shape) != 3:
            mask = label2rgb(mask, bg_label=0)
            mask_aug = label2rgb(augmented['mask'], bg_label=0)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask_aug = cv2.cvtColor(augmented['mask'], cv2.COLOR_BGR2RGB)

        ax[0, 0].imshow(image)
        ax[0, 0].set_title('Original image')

        ax[0, 1].imshow(image_aug)
        ax[0, 1].set_title('Augmented image')

        ax[1, 0].imshow(mask, interpolation='nearest')
        ax[1, 0].set_title('Original mask')

        ax[1, 1].imshow(mask_aug, interpolation='nearest')
        ax[1, 1].set_title('Augmented mask')

    f.tight_layout()

    if filename is not None:
        f.savefig(filename)

    return augmented['image'], augmented['mask']


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
