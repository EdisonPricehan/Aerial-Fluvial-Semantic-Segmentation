. ./00.run_vars.sh

echo "==== Inference ===="

# Inference
make \
     AFID_PATH=../dataset/afid/unified_felix \
     CKPT=${PATH_CKPT} \
     PROJECT_NAME="inference-${MODEL_ID}" \
     networks.inference

