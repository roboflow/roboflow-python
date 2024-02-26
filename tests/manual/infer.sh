#!/bin/env bash

export ROBOFLOW_CONFIG_DIR=./data/.config
python ../../roboflow/roboflowpy.py infer -w model-eval -m soccer/3 data/yellow/snacks.jpg
