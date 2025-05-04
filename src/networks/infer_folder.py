import os
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from networks.infer_trt import get_engine_context, trt_inference, post_process_mask
from networks.dataset import FluvialDataset
from utils.custom_transforms import get_transform_by_resolution_level
from utils.visualizer import show_compare


class InferFolder:
    def __init__(self, engine_path: str, dataset_path: str):
        self.engine_path = engine_path
        self.dataset_path = dataset_path
        self.res_level = 'low'

        # Init fluvial dataset
        res_tf = get_transform_by_resolution_level(self.res_level)
        self.dataset = FluvialDataset(dataset_path=dataset_path, transform=res_tf, target_transform=res_tf)

        # Fix user name mismatch between PC and Jetson
        self.dataset.img_files = [i.replace('edison', 'orin-nano') for i in self.dataset.img_files]
        self.dataset.mask_files = [i.replace('edison', 'orin-nano') for i in self.dataset.mask_files]
        self.dataset_len: int = len(self.dataset)
        print(f'{self.dataset_len=}')

    def run(self):
        try:
            model_load_start_time = time.time()

            # Load trt engine and context
            engine, context = get_engine_context(engine_path=self.engine_path)
            img, _ = self.dataset[0]  # Assume all data are of the same shape
            img = img.numpy()[None]  # (1, C, H, W)
            context.set_binding_shape(0, tuple(img.shape))

            infer_start_time = time.time()
            loading_time = infer_start_time - model_load_start_time
            print(f'Model loading time: {loading_time:.2f} s')

            for idx in tqdm(range(len(self.dataset))):
                # Get data
                img, mask = self.dataset[idx]
                img, mask = img.numpy(), mask.numpy()

                # Infer
                trt_outputs = trt_inference(engine=engine, context=context, data=img)
                # print(f'{type(trt_outputs)=} {type(trt_outputs[1])=}')

                # Post-process mask
                pred_mask = post_process_mask(trt_outputs[1], return_tensor=False)
                # print(f'{img.shape=} {mask.shape=} {pred_mask.shape=}')

                # Visualize (Disable to get correct inference time)
                show_compare(rgb=img.transpose((1, 2, 0)), gt_mask=mask, pred_mask=pred_mask)

            infer_end_time = time.time()
            infer_time = infer_end_time - infer_start_time
            print(f'Total infer time: {infer_time:.2f} s, average infer time: {infer_time / self.dataset_len:.2f} s')
        except Exception as e:
            print(e)



if __name__ == '__main__':
    # Define paths
    engine_path: str = '../models/unet-resnet34-128x128-fp16.trt'
    dataset_path: str = '../dataset/afid/dataset.csv'

    # Absolute paths
    engine_path = os.path.join(os.path.dirname(__file__), engine_path)
    dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)

    # Init inference
    infer_folder = InferFolder(engine_path=engine_path, dataset_path=dataset_path)

    # Start inference
    infer_folder.run()


