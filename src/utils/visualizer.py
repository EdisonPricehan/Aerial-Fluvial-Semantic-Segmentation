#!E:\anaconda/python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.ticker import PercentFormatter, FormatStrFormatter
from skimage.color import label2rgb
import os
import cv2
import fire
import numpy as np

from build_dataset import get_dataset_list


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
    plt.tight_layout()
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


def pie_chart_pixel(statistics_path, save_fig=True):
    """
    Plot a pie chart of the pixel statistics.
    :param statistics_path: relative path to the statistics file
    :param save_fig: whether to save the figure
    :return:
    """
    # get absolute path of statistics file
    stat_path_abs = os.path.join(os.path.dirname(__file__), statistics_path)
    if not os.path.exists(stat_path_abs):
        raise FileNotFoundError(f'Statistics file {stat_path_abs} not found')

    # read data from statistics file
    pixel_counts = get_dataset_list(stat_path_abs)
    if len(pixel_counts) < 2:
        raise ValueError('Not enough data to plot a pie chart')

    labels = pixel_counts[0]
    pixel_counts = pixel_counts[1:]
    pixel_counts = np.array([list(map(int, i)) for i in pixel_counts])
    print(f"{labels=}")
    total_pixels = np.sum(pixel_counts).astype(np.uint32)  # use uint32 to avoid overflow
    print(f"Analysing {len(pixel_counts)} images, {total_pixels} total pixels ...")
    labels_ratio = [np.sum(pixel_counts[:, l]) / total_pixels for l in range(len(labels))]
    print(f"{labels_ratio=}")

    # plot pie chart
    wedges, texts = plt.pie(labels_ratio, shadow=True, startangle=0, radius=1.2,
                            colors=['purple', 'green', 'yellow', 'blue', 'gray', 'brown', 'cyan', 'red'])

    legend_labels = ['{0} - {1:1.2f} %'.format(i, j * 100) for i, j in zip(labels, labels_ratio)]
    plt.legend(wedges, legend_labels, title="Pixel Labels", loc="center left", bbox_to_anchor=(-0.75, 0.5), fontsize=10)

    # plt.title('Labelled Pixel Distribution', loc="center", fontsize=12)
    if save_fig:
        # save figure
        fig_path = os.path.join(os.path.dirname(__file__), '../images', 'pie_chart_pixel.png')
        plt.savefig(fig_path, bbox_inches='tight')
    else:
        plt.show()


def histogram_pixel(statistics_path, save_fig=True):
    """
    plot histogram of water pixel ratio
    :param statistics_path: relative path to the statistics file
    :param save_fig: save the figure or not
    :return:
    """
    # get absolute path of statistics file
    stat_path_abs = os.path.join(os.path.dirname(__file__), statistics_path)
    if not os.path.exists(stat_path_abs):
        raise FileNotFoundError(f'Statistics file {stat_path_abs} not found')

    # read data from statistics file
    pixel_counts = get_dataset_list(stat_path_abs)
    if len(pixel_counts) < 2:
        raise ValueError('Not enough data to plot a pie chart')
    labels = pixel_counts[0]
    pixel_counts = pixel_counts[1:]
    pixel_counts = np.array([list(map(int, i)) for i in pixel_counts])  # convert string to int
    print(f"{labels=}")

    water_ratio = [cnts[0] / sum(cnts) for cnts in pixel_counts]
    # obstacle_ratio = [cnts[5] / sum(cnts) for cnts in pixel_counts]

    # plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    N_bins = 50
    n, bins, patches = ax.hist(water_ratio, bins=N_bins, weights=np.ones(len(water_ratio)) / len(water_ratio),
                                label='Water', linewidth=0.2, alpha=0.8)
    # print(f"{bins=}")
    # print(f"{n=}")

    # do bar color gradient
    for i in range(len(patches)):
        # move the desired color span from [0, 1] to [0.3, 0.7] to make the bar more visible
        color_percentile = (i / N_bins) * 0.7 + 0.3
        patches[i].set_facecolor(plt.cm.get_cmap("BuPu")(color_percentile))  # Blue to purple

    # plt.title('Water Pixels Image-wise Percentage Distribution Histogram')
    plt.xlabel('Water Pixels Image-wise Percentage', fontsize=12)
    plt.ylabel('Image Count Percentage', fontsize=12)

    # emphasize several percentile bins
    percentiles2ratio = {0.25: sum([n[i] for i, v in enumerate(bins[:-1]) if v >= 0.25]),
                         0.5: sum([n[i] for i, v in enumerate(bins[:-1]) if v >= 0.5]),
                         0.75: sum([n[i] for i, v in enumerate(bins[:-1]) if v >= 0.75])}
    print(f"{percentiles2ratio=}")
    # plot vertical lines at the desired percentiles and label them
    for p, v in percentiles2ratio.items():
        ax.axvline(p, color='green', linestyle='--', ymax=0.8, linewidth=2, label=f'{v:.2f}%')
        ax.text(p, 0.055, f'{v * 100:.2f}%', color='black', fontsize=14)
        ax.text(p, 0.05, f'>{p *100:.2f}%', color='black', fontsize=14)

    # make both x and y axes have percentage (%) scale
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1))

    # save histogram as figure or show it
    if save_fig:
        fig_path = os.path.join(os.path.dirname(__file__), '../images', 'histogram_pixel.png')
        plt.savefig(fig_path, bbox_inches="tight")  # need to make it tight to force text visibility
    else:
        plt.show()


def add_axis_names(image_path: str):
    assert image_path != '', "Image path is empty!"

    img = mpimg.imread(image_path)
    print(type(img))
    print(img.shape)
    h, w, _ = img.shape

    plt.imshow(img)
    plt.axis('off')

    plt.text(w * 0.48, h * 0.99, 'Epoch', fontsize=10, color='gray', weight='bold')
    plt.text(w * 0.02, h * 0.75, 'Validation F1 Score', fontsize=10, rotation='vertical', color='gray', weight='bold')

    # plt.show()
    plt.savefig(os.path.dirname(image_path) + '/new.png', bbox_inches="tight", dpi=300)


def double_sided_bars(csv_file_path: str):
    df = pd.read_csv(csv_file_path)
    print(df)

    # sns.barplot(data=df, x='arch', y="Params Mb", hue='encoder_name')

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.labelweight"] = "bold"
    fig, axes = plt.subplots(figsize=(10, 5), ncols=2, sharey=True)

    colors = ['red'] * 4 + ['green'] * 3 + ['blue'] * 5

    # axes[0].barh(df['Name'], df['Params Mb'], align='center', color=colors, zorder=10)
    # axes[0].set_title('Params Mb', fontsize=18, pad=15)
    # axes[1].barh(df['Name'], df['GFLOPs'], align='center', color=colors, zorder=10)
    # axes[1].set_title('GFLOPs', fontsize=18, pad=15)

    axes[0].barh(df['Name'], df['F1 Score'], align='center', color=colors, zorder=10)
    axes[0].set_title('F1 Score', fontsize=18, pad=15)
    axes[0].set_xlim([0.98, 0.9875])
    axes[0].xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    axes[1].barh(df['Name'], df['mIOU'], align='center', color=colors, zorder=10)
    axes[1].set_title('mIOU', fontsize=18, pad=15)
    axes[1].set_xlim([0.95, 0.963])
    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))

    axes[0].invert_xaxis()

    fig.tight_layout()

    # plt.show()

    fig.savefig('bar_metrics.png', bbox_inches="tight", dpi=300)

    # fig.savefig('bar_params.png', bbox_inches="tight", dpi=300)


if __name__ == '__main__':
    # fire.Fire()

    #### example usage 1 ####
    # python visualizer.py
    # histogram_pixel
    # '../dataset/Wabash-Wildcat/statistics.csv'
    #### example usage 1 ####

    #### example usage 2 ####
    # python visualizer.py
    # pie_chart_pixel
    # '../dataset/Wabash-Wildcat/statistics.csv'
    #### example usage 2 ####

    img_path = '../images/wandb/arch-merge.png'
    # img_path = '../images/wandb/encoder-merge.png'
    add_axis_names(img_path)

    # generate bar plot from csv file
    # csv_file = 'wandb_multicolumn.csv'
    # double_sided_bars(csv_file)


