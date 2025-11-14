make \
     ENCODER=se_resnext50_32x4d \
     DECODER=DeepLabV3Plus \
     ENCODER_WEIGHTS=imagenet \
     ACTIVATION=tanh \
     OPTIMIZER=adamax \
     LEARNING_RATE=0.0001 \
     EARLY_STOP_PATIENCE=10 \
     AFID_PATH_TRAIN=../dataset/afid/unified_felix/train.csv \
     AFID_PATH_TEST=../dataset/afid/unified_felix/test.csv \
     networks.custom.train
