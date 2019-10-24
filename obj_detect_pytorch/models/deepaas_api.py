# -*- coding: utf-8 -*-
"""
Model description
"""

import argparse
import pkg_resources
import os
# import project's config.py
import obj_detect_pytorch.config as cfg
import torchvision
from PIL import Image
import obj_detect_pytorch.models.model_utils as mutils
#import obj_detect_pytorch.models.create_resfiles as resfiles 

def get_metadata():
    """
    Function to read metadata
    """

    module = __name__.split('.', 1)

    pkg = pkg_resources.get_distribution(module[0])
    meta = {
        'Name': None,
        'Version': None,
        'Summary': None,
        'Home-page': None,
        'Author': None,
        'Author-email': None,
        'License': None,
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if line.startswith(par+":"):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta


def predict_file(*args):
    """
    Function to make prediction on a local file
    """
    message = 'Not implemented in the model (predict_file)'
    return message


def predict_data(*args):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    COCO_INSTANCE_CATEGORY_NAMES = mutils.category_names()

    outputpath=args[0]["outputpath"]
    threshold= float(args[0]["threshold"])
    thefile= args[0]['files'][0]
    thename= thefile.filename
    thepath= outputpath + "/" +thename
    thefile.save(thepath)    
    img = Image.open(thepath) 
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) 
    img = transform(img) 
    

    # Clear possible pre-existing sessions.
    backend.clear_session()
    pred = model([img]) 
    prediction_results["prediction"].update(prediction)
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] 
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] 
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] 
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]
              
    # Build result file and stream it back
    #result_image = resfiles.merge_images()
    #result_pdf = resfiles.create_pdf(result_image,prediction_results["prediction"])

    #return flask.send_file(filename_or_fp=result_pdf,
                           #as_attachment=True,
                           #attachment_filename=os.path.basename(result_pdf))
    
    message = mutils.format_prediction(pred_boxes,pred_class, pred_score)

    return message


def predict_url(*args):
    """
    Function to make prediction on a URL
    """
    message = 'Not implemented in the model (predict_url)'
    return message


###
# Uncomment the following two lines
# if you allow only authorized people to do training
###
#import flaat
#@flaat.login_required()
def train(train_args):
    """
    Train network
    train_args : dict
        Json dict with the user's configuration parameters.
        Can be loaded with json.loads() or with yaml.safe_load()    
    """

    run_results = { "status": "Not implemented in the model (train)",
                    "train_args": [],
                    "training": [],
                  }

    run_results["train_args"].append(train_args)

    print(run_results)
    return run_results


def get_train_args():
    """
    Returns a dict of dicts to feed the deepaas API parser
    """
    train_args = cfg.train_args

    # convert default values and possible 'choices' into strings
    for key, val in train_args.items():
        val['default'] = str(val['default']) #yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]

    return train_args

# !!! deepaas>=0.5.0 calls get_test_args() to get args for 'predict'
def get_test_args():
    predict_args = cfg.predict_args

    # convert default values and possible 'choices' into strings
    for key, val in predict_args.items():
        val['default'] = str(val['default'])  # yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]
        print(val['default'], type(val['default']))

    return predict_args

# during development it might be practical 
# to check your code from the command line
def main():
    """
       Runs above-described functions depending on input parameters
       (see below an example)
    """

    if args.method == 'get_metadata':
        get_metadata()       
    elif args.method == 'train':
        train(args)
    else:
        get_metadata()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')

    # get arguments configured for get_train_args()
    train_args = get_train_args()
    for key, val in train_args.items():
        parser.add_argument('--%s' % key,
                            default=val['default'],
                            type=type(val['default']),
                            help=val['help'])

    parser.add_argument('--method', type=str, default="get_metadata",
                        help='Method to use: get_metadata (default), \
                        predict_file, predict_data, predict_url, train')
    args = parser.parse_args()

    main()
