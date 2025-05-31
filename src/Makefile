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
	@echo "-----------------------------------------------------------------------------------------"
	@echo "             Robust Semantic Segmentation with Adversarial Feature Inference"
	@echo "-----------------------------------------------------------------------------------------"
	@echo ""
	@echo " make install                                          # install requirements"
	@echo " make AFID_PATH=[../dataset/afid/] networks.train      # train neural network"
	@echo " make CKPT=<checkpoint> networks.inference             # run inference on trained model"
	@echo ""

# Default paths for the AFID dataset and the checkpoint
AFID_PATH ?= ../dataset/afid/
# It defaults to this checkpoint, but it's not guaranteed to exist o be the best one.
CKPT ?= ../logs/lightning_logs/5ai3ighy/checkpoints/epoch=36-step=407.ckpt

install:
	@echo "Installing requirements..."
	pip install -r ../requirements.txt

networks.train:
	python -m networks.train $(AFID_PATH)/test.csv $(AFID_PATH)/train.csv

networks.inference:
	python -m networks.inference $(AFID_PATH)/test.csv $(CKPT)

.PHONY: all networks.train networks.inference install
