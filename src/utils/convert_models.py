import torch
import os

from networks.model import LitSegModel as LSM
from utils.image_transforms import resolution_mapping, supported_resolution_strings


__all__ = ['dummy_input', 'ckpt2pth', 'pth2onnx']


def dummy_input(res_level: str) -> torch.Tensor:
    assert res_level in resolution_mapping.keys(), f'{res_level} is not supported. Currently support {supported_resolution_strings()}.'

    h, w = resolution_mapping[res_level]

    dummy_image = torch.zeros((1, 3, h, w))

    return dummy_image


def ckpt2pth(ckpt_path: str, pth_name: str = 'model.pth') -> None:
    assert os.path.exists(ckpt_path), f'{ckpt_path} does not exist!'
    assert pth_name != '' and os.path.splitext(os.path.basename(pth_name))[-1] == '.pth'

    # Load checkpoint lightning model
    model: LSM = LSM.load_from_checkpoint(ckpt_path)
    model.eval()

    # Load state dict
    state_dict = model.model.state_dict()

    # Save the pytorch model
    target_dir: str = os.path.dirname(ckpt_path)
    target_pth_path: str = os.path.join(target_dir, pth_name)
    torch.save(state_dict, target_pth_path)


def pth2onnx(pth_path: str, onnx_name: str = 'model.onnx') -> None:
    assert os.path.exists(pth_path), f'{pth_path} does not exist!'
    assert onnx_name != '' and os.path.splitext(os.path.basename(onnx_name))[-1] == '.onnx'

    # Define pth model
    model = LSM(arch='Unet', encoder_name='resnet34', in_channels=3, out_classes=1)

    # Load state_dict from pth model
    state_dict = torch.load(pth_path)

    # Populate model with loaded state_dict
    model.model.load_state_dict(state_dict)
    model.eval()

    # Get dummy input
    dummy_image = dummy_input(res_level='low')
    assert dummy_image.dim() == 4, f'Dummy input dimension must be 4!'

    # Export the model to ONNX format
    target_dir: str = os.path.dirname(pth_path)
    target_onnx_path: str = os.path.join(target_dir, onnx_name)
    torch.onnx.export(
        model,
        dummy_image,
        target_onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )


if __name__ == '__main__':
    '''
    Convert checkpoint to pure pytorch form
    '''
    ckpt_path: str = '../models/unet-resnet34-128x128.ckpt'
    ckpt2pth(ckpt_path=ckpt_path, pth_name='unet-resnet34-128x128.pth')

    '''
    Convert pth to onnx form
    '''
    pth_path: str = '../models/unet-resnet34-128x128.pth'
    pth2onnx(pth_path=pth_path, onnx_name='unet-resnet34-128x128-explicit.onnx')


