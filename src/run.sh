# Training
make \
     ENCODER=se_resnext50_32x4d \
     DECODER=DeepLabV3Plus \
     ENCODER_WEIGHTS=imagenet \
     ACTIVATION=tanh \
     OPTIMIZER=adamax \
     LEARNING_RATE=0.0001 \
     EARLY_STOP_PATIENCE=10 \
     LOSS_TYPE=weighted_dice \
     AFID_PATH_TRAIN=../dataset/afid/unified_felix/train.csv \
     AFID_PATH_TEST=../dataset/afid/unified_felix/test.csv \
     networks.custom.train

MODEL_ID=`ls -lrt logs/lightning_logs | cut -c46-55 | tail -n 1`
MODEL_CKPT=`ls -l logs/lightning_logs/${MODEL_ID}/checkpoints/ | cut -c52-2000 | tail -n 1`
PATH_CKPT=../logs/lightning_logs/${MODEL_ID}/checkpoints/${MODEL_CKPT}

echo "==== Variables ===="
echo "MODEL_ID: ${MODEL_ID}"
echo "MODEL_CKPT: ${MODEL_CKPT}"
echo "PATH_CKPT: ${PATH_CKPT}"
echo ""


echo "==== Inference ===="

# Inference
make \
     AFID_PATH=../dataset/afid/unified_felix \
     CKPT=${PATH_CKPT} \
     PROJECT_NAME="inference-${MODEL_ID}" \
     networks.inference

echo "==== Video Inference ===="

mkdir -p /Users/felix/github/up/purdue/videos/wabash/output/${MODEL_ID}/

# Video Inference
time make \
     VIDEO_CSV=/Users/felix/github/up/purdue/Aerial-Fluvial-Semantic-Segmentation/src/dataset/afid/unified_felix/videos.csv \
     OUTPUT_VIDEO=/Users/felix/github/up/purdue/videos/wabash/output/${MODEL_ID}/video-${MODEL_ID}.mp4 \
     VIDEO_FPS=2 \
     MODEL_PATH=${PATH_CKPT} \
     networks.video

# Results
echo "==== Results ===="
echo ""
echo "Model ID: ${MODEL_ID}"
echo "Model CKPT: ${MODEL_CKPT}"
echo "Path CKPT: ${PATH_CKPT}"
echo ""
echo "Wandb: https://wandb.ai/felixcuello-universidad-de-palermo/lightning_logs/runs/${MODEL_ID}"
echo "Video: /Users/felix/github/up/purdue/videos/wabash/output/${MODEL_ID}/video-${MODEL_ID}.mp4"
echo ""
