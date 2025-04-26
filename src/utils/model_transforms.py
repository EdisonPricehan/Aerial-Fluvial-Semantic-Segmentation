
import os

import torch.onnx
import onnx
import onnxruntime

from src.networks.model import LitSegModel as SS_Model


def ckpt2onnx(ckpt_path, onnx_path, input_shape=(1, 3, 320, 544)):
    """
    Converts a PyTorch model checkpoint to ONNX format.
    """
    x = torch.randn(input_shape, requires_grad=True)
    model = SS_Model.load_from_checkpoint(ckpt_path).to('cuda')
    model.eval()
    model.to_onnx(file_path=onnx_path, input_sample=x.to('cuda'), export_params=True)
    assert os.path.isfile(onnx_path)


def check_onnx(onnx_path):
    """
    Checks the ONNX model for errors.
    """
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


if __name__ == '__main__':
    relative_path = '../logs/WildcatCreek-Data/1q0r6fkz/checkpoints/epoch=9-step=1759.ckpt'
    model_path = os.path.join(os.path.dirname(__file__), relative_path)
    output_path = os.path.join(os.path.dirname(model_path), 'model.onnx')

    # convert model to ONNX
    ckpt2onnx(model_path, output_path)

    # check ONNX model for errors
    # onnx_model_path = os.path.join(os.path.dirname(model_path), 'model.onnx')
    # check_onnx(onnx_model_path)
