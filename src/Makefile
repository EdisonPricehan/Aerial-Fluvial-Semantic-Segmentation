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
	@echo "----------------------------------------------------------------------------------------------------------------------"
	@echo "                            Robust Semantic Segmentation with Adversarial Feature Inference"
	@echo "----------------------------------------------------------------------------------------------------------------------"
	@echo ""
	@echo " make install                                                                      # install requirements"
	@echo " make AFID_PATH=[../dataset/afid/] LEARNING_RATE=[0.0001] networks.train           # train neural network"
	@echo " make AFID_PATH=[../dataset/afid/] LEARNING_RATE=[0.0001] networks.train.vgg13     # train VGG13 neural network"
	@echo " make CKPT=<checkpoint> networks.inference                                         # run inference on trained model"
	@echo " make INPUT_DIR=<input> OUTPUT_DIR=<output> png2jpeg                               # convert PNG files to JPEG"
	@echo " make INPUT_DIR=<input> OUTPUT_DIR=<output> rgb2grayscale                          # convert RGB PNG to grayscale PNG"
	@echo ""

# Default paths for the AFID dataset and the checkpoint
AFID_PATH ?= ../dataset/afid/
# It defaults to this checkpoint, but it's not guaranteed to exist o be the best one.
CKPT ?= ../logs/lightning_logs/5ai3ighy/checkpoints/epoch=36-step=407.ckpt
# Learning rate for training
LEARNING_RATE ?= 0.0001

install:
	@echo "Installing requirements..."
	pip install -r ../requirements.txt

networks.train:
	python -m networks.train -l $(LEARNING_RATE) $(AFID_PATH)/test.csv $(AFID_PATH)/train.csv

networks.train.vgg13:
	python -m networks.train -l $(LEARNING_RATE) --encoder vgg13 $(AFID_PATH)/test.csv $(AFID_PATH)/train.csv

networks.inference:
	python -m networks.inference $(AFID_PATH)/test.csv $(CKPT)

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

.PHONY: all networks.train networks.inference install png2jpeg rgb2grayscale
