from networks.model import LitSegModel as LSM, check_encoder_existence, check_decoder_existence

import numpy as np
from typing import Tuple
import os
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize


# Change matmul precision based on preference-overhead tradeoff
torch.set_float32_matmul_precision('medium')  # {'highest', 'high', 'medium'}


class InferPth:
    def __init__(self, model_path: str, arch: str = 'Unet', encoder: str = 'resnet34', device: str = 'cuda:0'):
        self.model_path = model_path
        self.arch = arch
        self.encoder = encoder
        self.device = device

        self.check_encoder_decoder()

        self.model = self.get_model(model_path)
        self.model.eval()

    def get_model(self, model_path: str) -> LSM:
        checkpoint_path = os.path.join(os.path.dirname(__file__), model_path)
        assert os.path.exists(checkpoint_path), f'{checkpoint_path} does not exist!'

        if checkpoint_path.endswith('.ckpt'):
            # PyTorch Lightning checkpoint
            return LSM.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=self.device)
        elif checkpoint_path.endswith('.pth'):
            # Standard PyTorch state_dict
            model = LSM(arch=self.arch, encoder_name=self.encoder, in_channels=3, out_classes=1)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.model.load_state_dict(state_dict)
            model.to(self.device)
            return model
        else:
            raise ValueError('Unsupported model file format.')

    def check_encoder_decoder(self):
        if not check_encoder_existence(self.encoder):
            raise ValueError(f"Encoder {self.encoder} is not supported.")
        if not check_decoder_existence(self.arch):
            raise ValueError(f"Architecture {self.arch} is not supported.")

    def infer(self, img_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference on a single image.
        :param img_path: Path to the input image.
        :return: Inference result as a numpy array.
        """

        img = read_image(img_path).to(self.device)
        img = Resize((128, 128))(img)  # Resize to match model input size
        print(f"Image shape: {img.shape}")

        img = img.unsqueeze(0)  # Add batch dimension

        mask = self.model.logits_to_mask(self.model(img), uint8=True)
        print(f'Mask shape: {mask.shape}')

        # Convert mask to numpy array and squeeze to remove batch dimension
        mask_np = mask.squeeze().cpu().numpy()
        img_np = img.squeeze().cpu().numpy()
        return img_np, mask_np


if __name__ == '__main__':
    # Define model and image paths
    model_path = '/home/edison/Research/Aerial-Fluvial-Semantic-Segmentation/src/models/unet-resnet34-128x128.pth'
    img_path = '/home/edison/Research/Aerial-Fluvial-Semantic-Segmentation/AerialFluvialDataset/WabashRiverDataset/wabash_dataset/images/wabash_downward_0000.jpg'

    inferer = InferPth(model_path=model_path)
    img, mask = inferer.infer(img_path=img_path)

    # Show image and mask
    import cv2
    # Convert to BGR for OpenCV
    img = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)  # Transpose to HWC for OpenCV
    cv2.imshow('Image', img)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
