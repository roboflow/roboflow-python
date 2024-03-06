#!/bin/env bash

export ROBOFLOW_CONFIG_DIR=./data/.config
# python ../../roboflow/roboflowpy.py import ./data/cultura-pepino-voc -w wolfodorpythontests -p cultura-pepino-upload-test-voc
# python ../../roboflow/roboflowpy.py import ./data/cultura-pepino-yolov8 -w wolfodorpythontests -p cultura-pepino-upload-test-yolov8
python ../../roboflow/roboflowpy.py import ./data/yellow -w wolfodorpythontests -p yellow-auto
# python ../../roboflow/roboflowpy.py import ./data/cultura-pepino-yolov5pytorch -w wolfodorpythontests -p cultura-pepino-upload-test-yolov5
# python ../../roboflow/roboflowpy.py import ./data/0311fisheye -w wolfodorpythontests -p 0311fisheye 
