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
                
                 'threshold': {'default': 0.8,
                               'help': 'Threshold of probability (0.0 - 1.0).',
                               'required': False
                         },
        
                 'output_type': {'default': "json",
                               'help': 'You can choose between "json" or "pdf" format.',
                               'required': False
                         },
                
                
                 'box_thickness': {'default': 2,
                               'help': 'Thickness of the box in pixels (Positive number starting from 1).',
                               'required': False
                         },
                
                 'text_size': {'default': 1,
                               'help': 'Size of the text in pixels (Positive number starting from 1).',
                               'required': False
                         },
                
                 'text_thickness': {'default': 2,
                               'help': 'Thickness of the text in pixels (Positive number starting from 1). ',
                               'required': False
                         },
}