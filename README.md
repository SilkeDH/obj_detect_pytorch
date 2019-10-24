DEEP Open Catalogue: Object Detection and Classification
==============================

[![Build Status](https://jenkins.indigo-datacloud.eu:8080/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/obj_detect_pytorch/master)](https://jenkins.indigo-datacloud.eu:8080/job/Pipeline-as-code/job/DEEP-OC-org/job/obj_detect_pytorch/job/master)

**Author:** Silke Donayre
**Project:** This work is part of the [DEEP Hybrid-DataCloud](https://deep-hybrid-datacloud.eu/) project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 777435.

This is a plug-and-play tool for object detection and classification using deep neural networks (Faster R-CNN ResNet-50 FPN Architecture) that were already pretrained on the [COCO Dataset](http://cocodataset.org/#home). The code uses the Pytorch Library, more information about it can be found at [Pytorch-Object-Detection](https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection). 

This module works on uploaded pictures and gives as ouput the rectangle coordinates x1,y1 and x2,y2 were the classificated object is located. It also provides you the probability of the classified object.

<p align="center">
<img src="./reports/figures/pytorchobj.png" alt="spectrogram" width="400">
</p>


Project Organization
------------

    ├── LICENSE
    ├── README.md              <- The top-level README for developers using this project.
    ├── data
    │   └── raw                <- The original, immutable data dump.
    │
    ├── docs                   <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── docker                 <- Directory for Dockerfile(s) for development
    │
    ├── models                 <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                             the creator's initials (if many user development), 
    │                             and a short `_` delimited description, e.g.
    │                             `1.0-jqp-initial_data_exploration.ipynb`.
    │
    ├── references             <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures            <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
    │                             generated with `pip freeze > requirements.txt`
    ├── test-requirements.txt  <- The requirements file for the test environment
    │
    ├── setup.py               <- makes project pip installable (pip install -e .) so obj_detect_pytorch can be imported
    ├── obj_detect_pytorch    <- Source code for use in this project.
    │   ├── __init__.py        <- Makes obj_detect_pytorch a Python module
    │   │
    │   ├── dataset            <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features           <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models             <- Scripts to train models and then use trained models to make
    │   │   │                     predictions
    │   │   └── deepaas_api.py
    │   │
    │   └── tests              <- Scripts to perfrom code testing + pylint script
    │   │
    │   └── visualization      <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini                <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
