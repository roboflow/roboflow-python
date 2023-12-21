#!/bin/env bash

export ROBOFLOW_CONFIG_DIR=./data/.config
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f coco -l ./data/cultura-pepino-coco
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f yolov5pytorch -l ./data/cultura-pepino-yolov5pytorch
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f yolov7pytorch -l ./data/cultura-pepino-yolov7pytorch
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f my-yolov6 -l ./data/cultura-pepino-my-yolov6
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f darknet -l ./data/cultura-pepino-darknet
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f voc -l ./data/cultura-pepino-voc
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f tfrecord -l ./data/cultura-pepino-tfrecord
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f createml -l ./data/cultura-pepino-createml
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f clip -l ./data/cultura-pepino-clip
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f multiclass -l ./data/cultura-pepino-multiclass
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f coco-segmentation -l ./data/cultura-pepino-coco-segmentation
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f yolo5-obb -l ./data/cultura-pepino-yolo5-obb
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f yolov8 -l ./data/cultura-pepino-yolov8
# python ../../roboflow/roboflowpy.py download motusbots/cultura-pepino/2 -f png-mask-semantic -l ./data/cultura-pepino-png-mask-semantic
python ../../roboflow/roboflowpy.py download gdit/aerial-airport/1 -f voc
