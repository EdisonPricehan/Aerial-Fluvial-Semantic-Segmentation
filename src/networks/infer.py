import os.path as osp
import numpy as np
import time
import torch
import torch.onnx
import torchvision
import torchvision.transforms as TF
import pytorch_lightning as pl
from PIL import Image
import segmentation_models_pytorch as smp
from SS_Model_Lit import SSModelGeneric as ssmg

# CUDA & TensorRT
# import pycuda.driver as cuda
from cuda import cuda
import pycuda.autoinit
import tensorrt as trt

# ONNX: pip install onnx, onnxruntime
# try:
#     import onnx
#     import onnxruntime as rt
# except ImportError as e:
#     raise ImportError(f'Please install onnx and onnxruntime first. {e}')


arch = 'Unet'  # PAN
encoder = 'resnet50'  # mit_b1

# image path
input_image_path = osp.join(osp.dirname(__file__),
                            '../images/AerialFluvialDataset/WildcatCreekDataset/wildcat_dataset/images/wildcat_forward_0118.jpg')
output_mask_path = osp.join(osp.dirname(__file__), '../images/predictions/{}-{}.png'.format(arch, encoder))

# model path
input_checkpoint_path = osp.join(osp.dirname(__file__), '../models/{}-{}.ckpt'.format(arch, encoder))
output_pth_path = osp.join(osp.dirname(__file__), '../models/{}-{}.pth'.format(arch, encoder))
input_pth_path = output_pth_path
output_onnx_path = osp.join(osp.dirname(__file__), '../models/{}-{}.onnx'.format(arch, encoder))
input_onnx_path = output_onnx_path
output_trt_path = osp.join(osp.dirname(__file__), '../models/{}-{}.trt'.format(arch, encoder))

height, width = 320, 544

device = torch.device('cuda')

TRT_LOGGER = trt.Logger()


def print_config():
    print('input_image_path: {}'.format(input_image_path))
    print('output_mask_path: {}'.format(output_mask_path))
    print('arch: {}'.format(arch))
    print('encoder: {}'.format(encoder))
    print('input_checkpoint_path: {}'.format(input_checkpoint_path))
    print('input_pth_path: {}'.format(input_pth_path))
    print('height: {} width: {}'.format(height, width))


def load_image(image_path: str = input_image_path):
    img = Image.open(image_path)

    transform = TF.Compose([
        TF.Resize(size=(height, width)),
        TF.PILToTensor(),
        TF.ConvertImageDtype(torch.float),
    ])

    img = transform(img)
    # img = img.to(device)
    img = img.unsqueeze(0)
    img = img.numpy().astype(np.float32)
    print('Loaded image size: {}'.format(img.shape))
    return img


def build_engine(onnx_model_path, tensorrt_engine_path, engine_precision='FP16', dynamic_axes=False,
                 img_size=(3, height, width), batch_size=1, min_engine_batch_size=1, opt_engine_batch_size=1,
                 max_engine_batch_size=1):
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
    if not osp.exists(onnx_model_path):
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
        profile.set_shape(inputTensor.name, (min_engine_batch_size, img_size[0], img_size[1], img_size[2]),
                          (opt_engine_batch_size, img_size[0], img_size[1], img_size[2]),
                          (max_engine_batch_size, img_size[0], img_size[1], img_size[2]))
        print('Set dynamic')
    else:
        profile.set_shape(inputTensor.name, (batch_size, img_size[0], img_size[1], img_size[2]),
                          (batch_size, img_size[0], img_size[1], img_size[2]),
                          (batch_size, img_size[0], img_size[1], img_size[2]))
    config.add_optimization_profile(profile)
    # network.unmark_output(network.get_output(0))

    # Write engine
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(tensorrt_engine_path, "wb") as f:
        f.write(engineString)


def trt_inference(engine, context, data):
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    print('nInput:', nInput)
    print('nOutput:', nOutput)

    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_dtype(i),
              engine.get_binding_dtype(i), engine.get_binding_dtype(i))
    for i in range(nInput, nInput + nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_dtype(i),
              engine.get_binding_dtype(i), engine.get_binding_dtype(i))

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


if __name__ == '__main__':
    print_config()

    '''
    Load input image
    '''
    img = load_image(input_image_path)

    '''
    Load checkpoint
    '''
    # checkpoint = torch.load(input_checkpoint_path)
    # checkpoint = pl.LightningModule.load_from_checkpoint(input_checkpoint_path)
    # checkpoint = ssmg.load_from_checkpoint(input_checkpoint_path)
    # model = ssmg.load_from_checkpoint(input_checkpoint_path)
    # print('Checkpoint loaded!')

    '''
    Load .pth model
    '''
    # model = ssmg(arch=arch, encoder_name=encoder, in_channels=3, out_classes=1)
    # model.load_state_dict(torch.load(input_pth_path, map_location='cuda:0'))
    # model.load_state_dict(torch.load(input_pth_path))
    # model.to(device)
    # print('pth model loaded!')

    # print('Hyper parameters: {}'.format(checkpoint['hyper_parameters']))
    # print('Checkpoint keys: {}'.format(checkpoint.keys()))

    '''
    Export to ONNX model
    '''
    # torch.onnx.export(model, img, output_onnx_path, opset_version=12, do_constant_folding=True, verbose=False)
    # print('ONNX model exported!')

    '''
    Start inferencetrt_outputs
    '''
    # model.eval()
    # with torch.no_grad():
    #     logits_mask = model(img)
    #     prob_mask = logits_mask.sigmoid()
    #     pred_mask = ((prob_mask > 0.5).float() * 255).to(torch.uint8).squeeze(0).to('cpu')
    #     print('Pred mask shape: {}'.format(pred_mask.shape))
    #     print('Inference finished!')
    #     torchvision.io.write_png(pred_mask, output_mask_path)
    #     print('Inferred mask saved!')

    # torch.save(model.state_dict(), output_pth_path)
    # print('Pth model saved!')

    # build TensorRT engine
    # build_engine(input_onnx_path, output_trt_path)
    # read the engine from file and deserialize
    with open(output_trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    # TensorRT inference
    context.set_binding_shape(0, (1, 3, height, width))
    trt_start_time = time.time()
    trt_outputs = trt_inference(engine, context, img)
    pred_mask = np.array(trt_outputs[1]).reshape(height, width)
    pred_mask_torch = torch.from_numpy(pred_mask)
    pred_mask_torch = pred_mask_torch.unsqueeze(0)
    pred_mask_torch = pred_mask_torch.sigmoid()
    pred_mask_torch = ((pred_mask_torch > 0.5).float() * 255).to(torch.uint8).to('cpu')
    print('Inference finished!')
    torchvision.io.write_png(pred_mask_torch, output_mask_path)
    print('Inferred mask saved!')
    print(pred_mask_torch[:10][:10])
    trt_end_time = time.time()
    print('--tensorrt--')
    # print(trt_outputs.shape)
    print('Time: ', trt_end_time - trt_start_time)
