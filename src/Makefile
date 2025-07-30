# -*- Makefile -*-
#
# Robust Semantic Segmentation with Adversarial Feature Inference
#
# The purpose behind this Makefile is to provide a simple interface for installing and
# running the code for the paper "Robust Semantic Segmentation with Adversarial Feature Inference".
#
# Since the CSV files have paths hardcoded, you will have to change the paths in the
# Makefile actions to match your local setup.
#

all:
	@echo "-------------------------------------------------------------------------------------------------------------------------"
	@echo "                            Robust Semantic Segmentation with Adversarial Feature Inference"
	@echo "-------------------------------------------------------------------------------------------------------------------------"
	@echo ""
	@echo " make install                                                                           # install requirements"
	@echo " make AFID_PATH=[../dataset/afid/] LEARNING_RATE=[0.0001] networks.train                # train neural network (original code)"
	@echo " make AFID_PATH=[../dataset/afid/] LEARNING_RATE=[0.0001] networks.train.vgg13          # train neural network (encoder=vgg13)"
	@echo " make AFID_PATH=[../dataset/afid/] LEARNING_RATE=[0.0001] networks.train.resnet110      # train neural network (encoder=resnet110)"
	@echo " make CKPT=<checkpoint> PROJECT_NAME=<project> networks.inference                       # run inference on trained model"
	@echo " make VIDEO_CSV=<csv> OUTPUT_VIDEO=<output> MODEL_PATH=<model> networks.video           # run video inference with model"
	@echo "                                                                                        # other parameters: [VIDEO_HEIGHT], [VIDEO_WIDTH], [VIDEO_FPS]"
	@echo " make INPUT_DIR=<input> OUTPUT_DIR=<output> png2jpeg                                    # convert PNG files to JPEG"
	@echo " make INPUT_DIR=<input> OUTPUT_DIR=<output> rgb2grayscale                               # convert RGB PNG to grayscale PNG"
	@echo ""
	@echo " Available Parameters:"
	@echo "   LEARNING_RATE=<float>     # Learning rate for training (default: 0.0001)"
	@echo "   ENCODER=<string>          # Encoder architecture (default: resnet34)"
	@echo "   DECODER=<string>          # Decoder architecture (default: DeepLabV3Plus)"
	@echo "   ENCODER_WEIGHTS=<string>  # Pre-trained encoder weights (default: imagenet)"
	@echo "   ACTIVATION=<string>       # Output activation function (default: sigmoid)"
	@echo "                             # Available: sigmoid, softmax, tanh"
	@echo "   OPTIMIZER=<string>        # Training optimizer (default: adam)"
	@echo "                             # Available: adam, sgd, rmsprop, adamw, adamax, adagrad"
	@echo ""
	@echo " Examples:"
	@echo "   make ACTIVATION=softmax networks.train                   # Train with softmax activation"
	@echo "   make ACTIVATION=tanh LEARNING_RATE=0.001 networks.train  # Train with tanh activation and custom learning rate"
	@echo "   make OPTIMIZER=sgd networks.train                        # Train with SGD optimizer"
	@echo "   make OPTIMIZER=adamw ACTIVATION=softmax networks.train   # Train with AdamW and softmax"
	@echo ""

# Default paths for the AFID dataset and the checkpoint
AFID_PATH ?= ../dataset/afid/
# It defaults to this checkpoint, but it's not guaranteed to exist o be the best one.
CKPT ?= ../logs/lightning_logs/5ai3ighy/checkpoints/epoch=36-step=407.ckpt
# Learning rate for training
LEARNING_RATE ?= 0.0001
# Project name for wandb logging during inference
PROJECT_NAME ?= aerial-fluvial-inference
# Encoder weights for training
ENCODER_WEIGHTS ?= imagenet
# Activation function for output layer
ACTIVATION ?= sigmoid
# Optimizer for training
OPTIMIZER ?= adam
# Video inference parameters (defaulting values)
VIDEO_HEIGHT ?= 320
VIDEO_WIDTH ?= 544
VIDEO_FPS ?= 10

install:
	@echo "Installing requirements..."
	pip install -r ../requirements.txt

networks.train:
	python -m networks.train -l $(LEARNING_RATE) --encoder $(ENCODER) --decoder $(DECODER) --activation $(ACTIVATION) --optimizer $(OPTIMIZER) $(AFID_PATH)/train.csv $(AFID_PATH)/test.csv

networks.train.vgg13:
	python -m networks.train -l $(LEARNING_RATE) --encoder vgg13 $(AFID_PATH)/train.csv $(AFID_PATH)/test.csv

networks.custom.train:
	python -m networks.train -l $(LEARNING_RATE) --encoder $(ENCODER) --decoder $(DECODER) --encoder-weights $(ENCODER_WEIGHTS) --activation $(ACTIVATION) --optimizer $(OPTIMIZER) $(AFID_PATH_TRAIN) $(AFID_PATH_TEST)

networks.inference:
	python -m networks.inference $(AFID_PATH)/test.csv $(CKPT) --project-name $(PROJECT_NAME)

networks.video:
	python -m networks.inference_video -i $(VIDEO_CSV) -o $(OUTPUT_VIDEO) -m $(MODEL_PATH) --height $(VIDEO_HEIGHT) --width $(VIDEO_WIDTH) -r $(VIDEO_FPS)

png2jpeg:
	@echo "Converting PNG files to JPEG..."
	@echo "Input directory: $(INPUT_DIR)"
	@echo "Output directory: $(OUTPUT_DIR)"
	python -m utils.convert_images png2jpeg $(INPUT_DIR) $(OUTPUT_DIR)

rgb2grayscale:
	@echo "Converting RGB PNG files to grayscale PNG..."
	@echo "Input directory: $(INPUT_DIR)"
	@echo "Output directory: $(OUTPUT_DIR)"
	python -m utils.convert_images rgb2grayscale $(INPUT_DIR) $(OUTPUT_DIR)

.PHONY: all networks.train networks.inference networks.video install png2jpeg rgb2grayscale
