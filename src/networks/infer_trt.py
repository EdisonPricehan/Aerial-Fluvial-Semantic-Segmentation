#!/usr/bin/env python

"""
Code in this script is for running inference on Nvidia Jetson, with minimal necessary dependencies.
"""

import torch
import os
import time
import numpy as np
from typing import Optional, Tuple
from PIL import Image

try:
    import tensorrt as trt
except ImportError as e:
    raise ImportError(f'Please make sure tensorrt is installed and can be found under "/usr/src/", '
                      f'and "/usr/lib/python3.10/dist-packages/ is in the PYTHONPATH.". '
                      f'If having "version GLIBCXX_3.4.30 not found" error, try upgrading the python version'
                      f'to the latest by "conda install -c conda-forge libstdcxx-ng", {e}.')

try:
    from cuda import cuda
except ImportError as e:
    raise ImportError(f'Please install cuda using "pip install cuda-python", {e}.')


# Image shape is hard coded here and needs to align with the trained segmentation model
height, width = 128, 128


def build_engine(
        onnx_model_path: str,
        tensorrt_engine_path: str,
        engine_precision='FP16',
        dynamic_axes=False,
        img_size=(3, height, width),
        batch_size=1,
        min_engine_batch_size=1,
        opt_engine_batch_size=1,
        max_engine_batch_size=1,
):
    """
    Please first try building the tensorrt engine using command line first, for example:
        trtexec --onnx=unet-resnet34-128x128.onnx --fp16 --saveEngine=unet-resnet34-128x128-fp16.trt
    If trtexec is not found, add an alias in .zshrc file:
        alias trtexec="/usr/src/tensorrt/bin/trtexec"
    If above commands do not work for you and you wanna build trt engine by code, refer to the following.
    """
    # Builder
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
    # Set FP16
    if engine_precision == 'FP16':
        config.set_flag(trt.BuilderFlag.FP16)

    # Onnx parser
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnx_model_path):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")

    with open(onnx_model_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    # Input
    inputTensor = network.get_input(0)
    # Dynamic batch (min, opt, max)
    print('inputTensor.name:', inputTensor.name)
    if dynamic_axes:
        profile.set_shape(inputTensor.name,
                          (min_engine_batch_size, img_size[0], img_size[1], img_size[2]),
                          (opt_engine_batch_size, img_size[0], img_size[1], img_size[2]),
                          (max_engine_batch_size, img_size[0], img_size[1], img_size[2]))
        print('Set dynamic')
    else:
        profile.set_shape(inputTensor.name,
                          (batch_size, img_size[0], img_size[1], img_size[2]),
                          (batch_size, img_size[0], img_size[1], img_size[2]),
                          (batch_size, img_size[0], img_size[1], img_size[2]))
    config.add_optimization_profile(profile)
    # network.unmark_output(network.get_output(0))

    # Write engine
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()

    with open(tensorrt_engine_path, "wb") as f:
        f.write(engineString)

    print("Succeeded building engine!")


def trt_inference(engine, context, data):
    """
    Low-level inference of trt engine.
    """
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    print('nInput:', nInput)
    print('nOutput:', nOutput)

    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_dtype(i))
    for i in range(nInput, nInput + nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_dtype(i))

    bufferH = []
    bufferH.append(np.ascontiguousarray(data.reshape(-1)))

    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))

    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cuda.cuMemAlloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cuda.cuMemcpyHtoD(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes)

    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cuda.cuMemcpyDtoH(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes)

    for b in bufferD:
        cuda.cuMemFree(b)

    return bufferH


def infer(engine_path: str, img_path: str, mask_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inference an image with trt engine, outputs the resized image and water mask.
    """
    assert os.path.exists(engine_path), f'{engine_path} does not exist!'

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Load image, resize to desired shape, then convert to numpy array
    img_arr = np.array(Image.open(img_path).resize((width, height)))
    print(f'After resize: {img_arr.shape=}')

    # Transpose numpy array to (C, H, W), increase the 1st dim, then normalize to range [0, 1]
    img_arr_input = img_arr.transpose((2, 0, 1))[None].astype(np.float32) / 255
    print(f'After transpose and normalization: {img_arr_input.shape=}')

    context.set_binding_shape(0, img_arr_input.shape)

    # s = context.get_binding_shape(0)
    # print(f'Binding input shape {s}.')

    # Do trt inference
    trt_start_time = time.time()
    trt_outputs = trt_inference(engine, context, img_arr_input)
    print(f'{trt_outputs=}')

    # Reshape to get inferred mask
    pred_mask = np.array(trt_outputs[1]).reshape(height, width)
    print(f'{pred_mask.shape=}')

    # Use pytorch's sigmoid function to convert to prob, then convert to uint mask
    pred_mask_torch = torch.from_numpy(pred_mask)
    pred_mask_torch = pred_mask_torch.sigmoid()
    pred_mask_torch = ((pred_mask_torch > 0.5).float() * 255).to(torch.uint8)
    print(f'{pred_mask_torch.shape=}')

    trt_end_time = time.time()
    print('Inference finished. Time cost: ', trt_end_time - trt_start_time)

    # Convert back to numpy for image display or save
    pred_mask_output = pred_mask_torch.detach().cpu().numpy()
    print(f'{pred_mask_output.shape=}')

    # Save mask to image
    if mask_path is not None:
        pred_mask_np_hwc = np.repeat(pred_mask_output[:, :, np.newaxis], 3, axis=2)
        mask = Image.fromarray(pred_mask_np_hwc)
        mask.save(mask_path)

    return img_arr_input, pred_mask_output


def validate_onnx(onnx_model_path: str):
    assert os.path.exists(onnx_model_path), f'{onnx_model_path} does not exist!'

    import onnx

    model = onnx.load(onnx_model_path)
    print([inp.name for inp in model.graph.input])


if __name__ == '__main__':
    # Example code to do semantic segmentation inference
    trt_model_path: str = '../models/unet-resnet34-128x128-fp16.trt'
    img_path: str = '../../AerialFluvialDataset/WildcatCreekDataset/wildcat_dataset/images/wildcat_downward_0000.jpg'
    # Convert to abs path
    trt_model_path = os.path.join(os.path.dirname(__file__), trt_model_path)
    img_path = os.path.join(os.path.dirname(__file__), img_path)
    # Define mask name and path
    img_dir_name: str = os.path.dirname(img_path)
    img_base_name: str = os.path.splitext(os.path.basename(img_path))[0]
    mask_name: str = f'{img_base_name}_mask.png'
    mask_path: str = os.path.join(img_dir_name, mask_name)
    mask_path = os.path.join(os.path.dirname(__file__), mask_path)

    infer(engine_path=trt_model_path, img_path=img_path, mask_path=mask_path)



