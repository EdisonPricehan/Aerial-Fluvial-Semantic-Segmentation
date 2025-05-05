import numpy as np
import torch
import torchvision
import os
import time
from tqdm import tqdm

from networks.infer_trt import get_engine_context, trt_inference, trt_inference_policy, post_process_mask
from networks.dataset import FluvialDataset, VideoDataset
from utils.image_transforms import get_transform_by_resolution_level
from utils.patchification import patch_array, inflate_patch_mask
import utils.video_transforms as VT
from utils.visualizer import show_compare_policy


class InferBoth:
    def __init__(
            self,
            perception_engine_path: str,
            policy_engine_path: str,
            dataset_path: str,
    ) -> None:
        """
        Infer both semantic segmentation engine and river following policy engine.
        """
        # Init variables
        self.perc_engine_path = perception_engine_path
        self.policy_engine_path = policy_engine_path
        self.dataset_path = dataset_path
        self.res_level = 'low'

        # Define constants
        self.height, self.width = 128, 128  # Image pixel dimension
        self.patch_height, self.patch_width = 8, 8  # Mask's single patch dimension

        # Init image-mask dataset or video dataset
        if 'video' in dataset_path:  # Init video dataset
            fps = 2
            duration = 100  # second
            self.dataset = VideoDataset(
                csv_file=dataset_path,
                transform=torchvision.transforms.Compose(
                    [
                        VT.VideoFilePathToTensor(fps=fps, max_len=duration * fps),
                        VT.VideoResize([self.height, self.width])
                    ]
                )
            )[0]
            self.dataset = self.dataset.permute(1, 0, 2, 3)
            print(f'{self.dataset.shape=}')
            # exit(0)
        else:  # Init fluvial dataset
            res_tf = get_transform_by_resolution_level(self.res_level)
            self.dataset = FluvialDataset(dataset_path=dataset_path, transform=res_tf, target_transform=res_tf)

            # Fix user name mismatch between PC and Jetson for the fluvial dataset
            self.dataset.img_files = [i.replace('edison', 'orin-nano') for i in self.dataset.img_files]
            self.dataset.mask_files = [i.replace('edison', 'orin-nano') for i in self.dataset.mask_files]

        self.dataset_len: int = len(self.dataset)
        print(f'{self.dataset_len=}')

        # Define engine and context for both perception and policy trt models
        self.perc_engine, self.perc_context = get_engine_context(self.perc_engine_path)
        self.policy_engine, self.policy_context = get_engine_context(self.policy_engine_path)
        print(f'Engines and contexts are created for both perception and policy.')

    def run(self) -> None:
        for idx in tqdm(range(len(self.dataset))):
            # Get image and ground-truth mask
            if 'video' in self.dataset_path:
                img = self.dataset[idx].numpy()
            else:
                img, mask = self.dataset[idx]
                img, mask = img.numpy(), mask.numpy()
            print(f'{img.shape=}')

            # Perception infer
            perc_trt_outputs = trt_inference(engine=self.perc_engine, context=self.perc_context, data=img)
            pred_mask = post_process_mask(perc_trt_outputs[1], return_tensor=False, h=self.height, w=self.width)
            print(f'{pred_mask.shape=}')

            # Patchify to get policy observation
            obs = patch_array(
                mask_array=pred_mask,
                is_uint8=True,
                patch_size_x=self.patch_height,
                patch_size_y=self.patch_width,
                patch_step=self.patch_width,
            )
            print(f'{obs.shape=}')
            reset_flag = np.array([[False]], dtype=bool)

            # Policy infer
            policy_trt_outputs = trt_inference_policy(
                engine=self.policy_engine,
                context=self.policy_context,
                inputs=[obs, reset_flag]
            )
            action = policy_trt_outputs[-1]
            print(f'{action=}')

            obs_inflated = inflate_patch_mask(
                obs=obs,
                image_size=self.height,
                patch_dim_x=16,
                patch_dim_y=16,
                patch_size_x=self.patch_height,
                patch_size_y=self.patch_width,
            )

            show_compare_policy(
                rgb=img.transpose((1, 2, 0)),  # (H, W, C)
                pred_mask=pred_mask,
                patchified_mask=obs_inflated,
                action=action,
            )

            print('*' * 50)


if __name__ == '__main__':
    # Define constants
    perception_trt_path: str = '../models/unet-resnet34-128x128-fp16.trt'
    policy_trt_path: str = '../models/policy.trt'
    # dataset_path: str = '../dataset/afid/dataset.csv'  # Image-Mask dataset
    dataset_path: str = 'video-path/video_path.csv'  # Video dataset

    # Init inference
    infer_both = InferBoth(
        perception_engine_path=perception_trt_path,
        policy_engine_path=policy_trt_path,
        dataset_path=dataset_path,
    )

    # Run inference on dataset
    infer_both.run()
