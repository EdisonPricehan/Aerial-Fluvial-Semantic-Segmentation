#!E:\anaconda/python

import os
import argparse
from pprint import pprint

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.io import write_video

from dataset import VideoDataset
from src.utils import video_transforms as transforms
from inference import get_model, inference

if __name__ == '__main__':
    # parse necessary arguments if run as script
    parser = argparse.ArgumentParser(description="Inference video, resize, and save as another video")
    parser.add_argument('-i', '--input', metavar='DIR', help='Relative path of input video(s) csv file')
    parser.add_argument('-o', '--output', metavar='DIR', help='Relative path of output video')
    parser.add_argument('-m', '--model-path', type=str, default='',
                        help='Relative path of model to be loaded if want to do inference')
    parser.add_argument('--height', type=int, default=320, help='Output video height')
    parser.add_argument('--width', type=int, default=544, help='Output video width')
    parser.add_argument('-r', '--rate', type=int, default=10, help='Desired output video fps')
    parser.add_argument('-d', '--duration', type=int, default=0, help='Duration in sec to infer, 0 means all video')

    # init args
    args = parser.parse_args()
    pprint(args.__dict__)

    # parse arguments as local variables
    input_video_csv_path = args.input
    output_video_path = args.output
    model_path = args.model_path
    img_height = args.height
    img_width = args.width
    desired_fps = args.rate
    duration = args.duration

    # change paths to absolute paths and check video existence
    input_video_csv_path = os.path.join(os.path.dirname(__file__), input_video_csv_path)
    assert os.path.exists(input_video_csv_path), f"{input_video_csv_path} does NOT exist!"
    print(f"{input_video_csv_path=}")
    output_video_path = os.path.join(os.path.dirname(__file__), output_video_path)

    # load model if needed
    model = None
    if model_path != '':
        assert model_path != '', "Model path is empty!"
        print("Start loading model ...")
        model_path = os.path.join(os.path.dirname(__file__), model_path)
        print(f"{model_path=}")
        model = get_model(model_path)
        print("Model loaded!")

    # create video dataset
    print("Creating video dataset ...")
    datasets = VideoDataset(input_video_csv_path,
                            transform=torchvision.transforms.Compose([
                                transforms.VideoFilePathToTensor(fps=desired_fps, max_len=duration * desired_fps),
                                transforms.VideoResize([img_height, img_width])]))
    print("Video dataset created!")

    # process all videos
    for video_tensor in datasets:
        # create video dataloader
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        print(f"Extracted video tensor shape: {video_tensor.shape}")
        video_loader = DataLoader(video_tensor, batch_size=1, shuffle=False)

        if model:  # feed into model and get predictions
            print("Start inferring ...")
            pred_masks_list = inference(model, video_loader)
            target_tensor = torch.cat(pred_masks_list, dim=0)  # [N x C x H x W]
            target_tensor = target_tensor.permute(0, 2, 3, 1)  # [N x H x W x C]
            target_tensor = torch.cat([target_tensor] * 3, dim=-1)  # single channel gray to 3 channel rgb gray
            print(f"target tensor shape: {target_tensor.shape}")
            print("Inference finished!")
        else:  # only get transformed tensor
            target_tensor = video_tensor.permute(0, 2, 3, 1)  # [N x H x W x C]
            target_tensor = (target_tensor * 255).to(torch.uint8)
            print("Got all video frames.")

        # save video
        print("Starting saving video ...")
        print(f"Written video tensor shape: {target_tensor.shape}")
        write_video(output_video_path, target_tensor, fps=desired_fps, video_codec='h264')
        print("Video saved!")

    print("Video inference finished!")
