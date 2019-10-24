# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))

DATA_DIR = path.join(BASE_DIR,'data') # Location of model data and output files

train_args = { 'arg1': {'default': 1,
                        'help': '',
                        'required': False
                        },
}

# !!! deepaas>=0.5.0 calls get_test_args() to get args for 'predict'
predict_args = { 'outputpath': {'default': "/tmp",
                                'help': 'Path for loading the model',
                                'required': True
                               },
                 'threshold': {'default': 0.5,
                               'help': 'Threshold of probability ',
                               'required': False
                         },
}