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
	@echo " make install                                                                      # install requirements"
	@echo " make AFID_PATH=[../dataset/afid/] LEARNING_RATE=[0.0001] networks.train           # train neural network (original code)"
	@echo " make AFID_PATH=[../dataset/afid/] LEARNING_RATE=[0.0001] networks.train.vgg13     # train neural network (encoder=vgg13)"
	@echo " make CKPT=<checkpoint> PROJECT_NAME=<project> networks.inference                  # run inference on trained model"
	@echo " make VIDEO_CSV=<csv> OUTPUT_VIDEO=<output> MODEL_PATH=<model> networks.video      # run video inference with model"
	@echo "                                                                                   # other parameters: [VIDEO_HEIGHT], [VIDEO_WIDTH], [VIDEO_FPS]"
	@echo " make INPUT_DIR=<input> OUTPUT_DIR=<output> png2jpeg                               # convert PNG files to JPEG"
	@echo " make INPUT_DIR=<input> OUTPUT_DIR=<output> rgb2grayscale                          # convert RGB PNG to grayscale PNG"
	@echo ""

# Default paths for the AFID dataset and the checkpoint
AFID_PATH ?= ../dataset/afid/
# It defaults to this checkpoint, but it's not guaranteed to exist o be the best one.
CKPT ?= ../logs/lightning_logs/5ai3ighy/checkpoints/epoch=36-step=407.ckpt
# Learning rate for training
LEARNING_RATE ?= 0.0001
# Project name for wandb logging during inference
PROJECT_NAME ?= aerial-fluvial-inference
# Video inference parameters (defaulting values)
VIDEO_HEIGHT ?= 320
VIDEO_WIDTH ?= 544
VIDEO_FPS ?= 10

install:
	@echo "Installing requirements..."
	pip install -r ../requirements.txt

networks.train:
	python -m networks.train -l $(LEARNING_RATE) $(AFID_PATH)/train.csv $(AFID_PATH)/test.csv

networks.train.vgg13:
	python -m networks.train -l $(LEARNING_RATE) --encoder vgg13 $(AFID_PATH)/train.csv $(AFID_PATH)/test.csv

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
