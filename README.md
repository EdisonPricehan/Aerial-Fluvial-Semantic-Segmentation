# Aerial Fluvial Image Dataset (AFID) for Semantic Segmentation
![](./src/images/showcase/afid_showcase.png)

## AFID Download
Please find the public Aerial Fluvial Image Dataset ([AFID](https://purr.purdue.edu/publications/4105/1)) on Purdue University Research Repository.
AFID contains manually annotated images shot by the drone over the Wabash River and the Wildcat Creek in Indiana.
Unzip the downloaded zip file and put *AerialFluvialDataset* folder into the *Aerial-Fluvial-Semantic-Segmentation* folder.
Also remember to unzip the *WildcatCreekDataset* and *WabashRiverDataset* in the *AerialFluvialDataset*.


## Dataset Creation
The [FluvialDataset](./src/networks/dataset.py) class accepts image and mask absolute path pairs in a csv file.
An example is the csv files in this [folder](./src/dataset/afid), containing the whole dataset, train set and test set.
These csv files can be generated using the function *build_csv_from_datasets* in [build_dataset.py](./src/utils/build_dataset.py).
Since the csv files store the absolute paths, it is recommended to rebuild these files after git cloning.


## Preparation
The training mainly depends on PyTorch Lightning and segmentation_models_pytorch.
A virtual python environment (e.g., miniconda) is recommended to test this repo while separating from your system python libraries.
```shell
conda create -n afid python==3.10
```
*afid* is the name of the virtual python environment. After activated this environment by `conda activate afid`,
ensure you have installed all dependencies by
```shell
pip install -r requirements.txt
```


## Training
The training code is in [train.py](./src/networks/train.py).
An example usage is
```shell
python -m networks.train '../dataset/afid/train.csv' '../dataset/afid/test.csv'
```
If having *No such file or directory* error, make sure the paths in *train.csv* and *test.csv* are pointing to the correct location of AFID data.


## Logging
We use [wandb](https://wandb.ai/home) for logging intermediate checkpoints.
You can visualize the training progress, and inspect the trained models from their website (although models are also stored locally, it is more convenient to see which model is the best on their website).


## Inference
The inference code is in [inference.py](./src/networks/inference.py).
An example usage is
```shell
python -m networks.inference '../dataset/afid/test.csv' '../models/unet-resnet34-128x128.ckpt'
```

## Prediction
If you want to do prediction on your own video, you can use the [inference_video.py](./src/networks/inference_video.py) script.
An example usage is
```shell
python -m networks.inference_video -i video-path/video_path.csv -o ./output.mp4 -m ../models/unet-resnet34-128x128.ckpt --height 128 --width 128 -r 1
```
where `-i` is the csv file path that contains all input videos, `-o` is the output video path with desired suffix, `-m` is the model checkpoint path, `--height` and `--width` are the height and width of the input frames, and `-r` is the frame rate of the output video.


## Citation
If you use the AFID dataset or this repo in your work, please cite our paper. Thanks.
```
@article{wang2023aerial,
  title={Aerial fluvial image dataset for deep semantic segmentation neural networks and its benchmarks},
  author={Wang, Zihan and Mahmoudian, Nina},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={16},
  pages={4755--4766},
  year={2023},
  publisher={IEEE}
}
```

## Training Diagram

```mermaid
graph TD
    A["Training Start"] --> B["Parse Arguments<br/>- train/valid CSV paths<br/>- encoder/decoder types<br/>- batch size, epochs, learning rate"]

    B --> C["Validate Architecture<br/>- Check encoder exists in SMP library<br/>- Check decoder exists in SMP library<br/>SMP = Segmentation Models PyTorch"]

    C --> D["Load Datasets<br/>FluvialDataset Class<br/>- Training data (images + masks)<br/>- Validation data (images + masks)<br/>- Apply transforms/augmentations"]

    D --> E["Build Model<br/>LitSegModel (PyTorch Lightning)<br/>- Architecture: decoder + encoder<br/>- Loss: Dice Loss (1 - Dice Coefficient)<br/>- Optimizer: Adam"]

    E --> F["Setup Logging & Callbacks<br/>- WandB Logger (Weights & Biases)<br/>- ModelCheckpoint (saves best F1)<br/>- EarlyStopping (patience=10 epochs)"]

    F --> G["Training Loop<br/>For each epoch (default: 75):"]

    G --> H["Training Phase"]
    H --> H1["For each batch:<br/>1. Forward pass through network<br/>2. Compute Dice Loss<br/>3. Backpropagation & gradient update<br/>4. Calculate confusion matrix:<br/>   • TP = True Positives<br/>   • FP = False Positives<br/>   • FN = False Negatives<br/>   • TN = True Negatives"]

    H1 --> I["Training Epoch End<br/>Calculate & Log Metrics:<br/>• IoU = Intersection over Union<br/>  - Per-image IoU (avg per image)<br/>  - Dataset IoU (aggregate)<br/>• F1 = 2×(Precision×Recall)/(Precision+Recall)<br/>• Accuracy = (TP+TN)/(TP+FP+FN+TN)"]

    I --> J["Validation Phase"]
    J --> J1["For each batch (no gradients):<br/>1. Forward pass through network<br/>2. Compute Dice Loss<br/>3. Calculate confusion matrix:<br/>   • TP = True Positives (water predicted as water)<br/>   • FP = False Positives (non-water predicted as water)<br/>   • FN = False Negatives (water predicted as non-water)<br/>   • TN = True Negatives (non-water predicted as non-water)"]

    J1 --> K["Validation Epoch End<br/>Calculate & Log Metrics:<br/>• valid_dataset_iou<br/>• valid_per_image_iou<br/>• valid_dataset_f1 (PRIMARY METRIC)<br/>• valid_dataset_accuracy<br/>Send to WandB dashboard"]

    K --> L{"Monitor Primary Metric<br/>valid_dataset_f1<br/>(Validation F1 Score)<br/>Is this the best so far?"}
    L -->|Yes| M["Save Best Model<br/>ModelCheckpoint saves .ckpt file<br/>Based on highest valid_dataset_f1<br/>Contains: weights, optimizer state, epoch"]
    L -->|No| N{"Early Stopping Check<br/>Has valid_dataset_f1<br/>improved in last 10 epochs?<br/>(patience=10)"}

    M --> N
    N -->|No improvement<br/>Stop training| O["Training Complete<br/>Best model saved as .ckpt<br/>Ready for inference/conversion<br/>Can be converted to ONNX/TensorRT"]
    N -->|Still improving<br/>Continue| G

    style A fill:#e1f5fe
    style O fill:#c8e6c9
    style E fill:#fff3e0
    style L fill:#fce4ec
    style N fill:#fce4ec
    style K fill:#f3e5f5
```
