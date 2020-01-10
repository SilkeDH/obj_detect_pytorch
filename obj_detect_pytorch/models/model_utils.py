# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#

import requests

def format_prediction(boxes, labels, probabilities):
    d = {
        "status": "ok",
        "predictions": [],
    }
    
    for i in range(len(boxes)):
        pred = {
            "label": labels[i],
            "probability": str(probabilities[i]),
            "rectangle Coodinates":{
                "coords": [{"Coordinates 1": str(boxes[i][0])},
                           {"Coordinates 2": str(boxes[i][1])}],
            },
        }
        d["predictions"].append(pred)
                         
    return d

def format_train(network, accuracy, nepochs, data_size, 
                 time_prepare, mn_train, std_train):


    train_info = {
        "network": network,
        "test accuracy": accuracy,
        "n epochs": nepochs,
        "train set (images)": data_size,
        "validation set (images)": data_size,
        "test set (images)": data_size,
        "time": {
                "time to prepare": time_prepare,
                "mean per epoch (s)": mn_train,
                "std (s)": std_train,
                },
    }

    return train_info

def category_names():
    
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    return COCO_INSTANCE_CATEGORY_NAMES
    