#!E:\anaconda/python

import os
import torch
import time
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, flop_count_table

from src.networks.SS_Model_Lit import SSModelGeneric as ssmg


if __name__ == '__main__':
    # specify the model checkpoint to load
    model_path = os.path.join(os.path.dirname(__file__),
                              '../logs/baseline/smquglle/checkpoints/epoch=8-step=2025.ckpt')
    model = ssmg.load_from_checkpoint(checkpoint_path=model_path)
    model.eval()

    # generate a dummy input rgb image
    input_tensor = torch.rand(1, 3, 320, 544) * 255

    # calculate number of flops and params
    flops = FlopCountAnalysis(model, input_tensor)
    print(flop_count_table(flops))

    # calculate average inference time
    total_duration = 0
    inference_times = 300
    with torch.no_grad():
        for i in tqdm(range(inference_times)):
            start = time.time()
            model(torch.rand(1, 3, 320, 544) * 255)
            total_duration += (time.time() - start)
    print(f"Average inference time: {total_duration / inference_times * 1000} milli-seconds.")

