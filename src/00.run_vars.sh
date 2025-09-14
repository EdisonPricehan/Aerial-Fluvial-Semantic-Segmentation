MODEL_ID=`ls -lrt logs/lightning_logs | awk '{ print $9 }' | tail -n 1`
MODEL_CKPT=`ls -lrt logs/lightning_logs/${MODEL_ID}/checkpoints/ | awk '{ print $9 }' | tail -n 1`
PATH_CKPT=../logs/lightning_logs/${MODEL_ID}/checkpoints/${MODEL_CKPT}

echo "==== Variables ===="
echo "MODEL_ID: ${MODEL_ID}"
echo "MODEL_CKPT: ${MODEL_CKPT}"
echo "PATH_CKPT: ${PATH_CKPT}"
echo ""
