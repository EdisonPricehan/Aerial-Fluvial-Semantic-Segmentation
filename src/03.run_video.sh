. ./00.run_vars.sh

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
