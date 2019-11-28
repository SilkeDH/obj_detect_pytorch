# -*- coding: utf-8 -*-
"""
Model description
"""
from webargs import fields
import argparse
import pkg_resources
import os
import obj_detect_pytorch.config as cfg
import torchvision
from PIL import Image
import obj_detect_pytorch.models.model_utils as mutils
import obj_detect_pytorch.models.model_metrics as mmetrics
import obj_detect_pytorch.models.create_resfiles as resfiles 
from fpdf import FPDF
import cv2
import ignite
from torchvision import transforms, utils



def get_metadata():
    #Metadata of the model:
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

def warm():
    """
    This is called when the model is loaded, before the API is spawned. 
    If implemented, it should prepare the model for execution. This is useful 
    for loading it into memory, perform any kind of preliminary checks, etc.
    """

###
# Uncomment the following two lines
# if you allow only authorized people to do training
###
#import flaat
#@flaat.login_required()
def get_train_args():
    #Training arguments:
    return {
        "arg1": fields.Str(
            required=False,  # force the user to define the value
            missing="foo",  # default value to use
            enum=["choice1", "choice2"],  # list of choices
            description="Argument one"  # help string
        ),
    }

def train(train_args):
    #Training of the model:
    run_results = { "status": "Not implemented in the model (train)",
                    "train_args": [],
                    "training": [],
                  }
    run_results["train_args"].append(train_args)
    return run_results
    

def get_predict_args():
    #Prediction arguments:
    return {
        "files": fields.Field(
            description="Data file to perform inference on.",
            required=True,
            type="file",
            location="form"),
        
        "outputpath": fields.Str(
            required=False,  # force the user to define the value
            missing="/temp",  # default value to use
            description="Specifies in which path the image should be stored."  # help string
        ),
        
        "outputtype": fields.Str(
            required=False,  # force the user to define the value
            missing="json",  # default value to use
            enum=["json", "pdf"],  # list of choices
            description="Specifies the output format."  # help string
        ),
        
        "threshold": fields.Str(
            required=False,  # force the user to define the value
            missing= 0.8,  # default value to use
            description="Threshold of probability (0.0 - 1.0)."  # help string
        ),
        
        "box_thickness": fields.Str(
            required=False,  # force the user to define the value
            missing= 2,  # default value to use
            description="Thickness of the box in pixels (Positive number starting from 1)."  # help string
        ),
        
        "text_size": fields.Str(
            required=False,  # force the user to define the value
            missing= 1 ,  # default value to use
            description="Size of the text in pixels (Positive number starting from 1)."  # help string
        ),
        
        "text_thickness": fields.Str(
            required=False,  # force the user to define the value
            missing= 2,  # default value to use
            description="Thickness of the text in pixels (Positive number starting from 1)."  # help string
        ),
     }

# during development it might be practical 
# to check your code from the command line
    
def predict_file(**args):
    message = 'Not implemented in the model (predict_file)'
    return message
 
def predict(**args):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    COCO_INSTANCE_CATEGORY_NAMES = mutils.category_names()  #Category names trained.
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) 
    #Reading the image and saving it.
    outputpath=args['outputpath'] 
    threshold= float(args['threshold'])
    thefile= args['files']
    img1 = Image.open(thefile.filename)  
    other_path = '{}/Input_image_patch.png'.format(cfg.DATA_DIR)
    img1.save(other_path)
    
    #Prediction and results.
    img1 = transform(img1) 
    pred = model([img1]) 
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] 
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] 
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] 
    pred_boxes = pred_boxes[:pred_t+1]   #Boxes.
    pred_class = pred_class[:pred_t+1]   #Name of the class.
    pred_score = pred_score[:pred_t+1]   #Prediction probability.
    
    #a = mmetrics.get_metrics() #Predict images of the classifier and get the metrics.
    
    if (args["outputtype"] == "pdf"):  
        #PDF Format:    
        #Drawing the boxes around the objects in the images + putting text + probabilities. 
        img_cv = cv2.imread(thepath) # Read image with cv2
        for i in range(len(pred_boxes)):
            cv2.rectangle(img_cv, pred_boxes[i][0], pred_boxes[i][1], color= (0,255,255) , 
                          thickness= int(args['box_thickness']))  # Draws rectangle.
            cv2.putText(img_cv,str(pred_class[i]) + " " + str(float("{0:.4f}".format(pred_score[i]))), pred_boxes[i][0],
                    cv2.FONT_HERSHEY_SIMPLEX, int(args['text_size']), (0,255,255),thickness= int(args['text_thickness'])) 
        class_path = '{}/Classification_map.png'.format(cfg.DATA_DIR)
        cv2.imwrite(class_path,img_cv)    
    
        #Merge original image with the classified one.
        result_image = resfiles.merge_images()

        #Create the PDF file.
        result_pdf = resfiles.create_pdf(result_image, pred_boxes, pred_class, pred_score)

        return flask.send_file(filename_or_fp=result_pdf,
                           as_attachment=True,
                           attachment_filename=os.path.basename(result_pdf))
                         
        message = 'Not implemented in the model (predict_file)'
        return message 
        
    else: 
        #JSON format:
        message = mutils.format_prediction(pred_boxes,pred_class, pred_score)  
        return message

def main():
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
                        predict_file, predict, train')
    args = parser.parse_args()

    main()
