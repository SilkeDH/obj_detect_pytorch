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
import obj_detect_pytorch.models.create_resfiles as resfiles 
from fpdf import FPDF
import cv2
import ignite
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import glob
import torch
import numpy as np
import pandas as pd


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

def predict_image(image):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    test_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                     ])
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

class COCO2017(Dataset):
    """COCO 2017 dataset."""

    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if json_file is not None:
            with open(json_file,'r') as COCO:
                js = json.loads(COCO.read())
                val_categories = json.dumps(js) 
                
        image_ids = []
        categ_ids = []
        #Get categories of the validation images and ids.
        for i in range(32800):
            image_id = json.dumps(js['annotations'][i]['image_id'])
            miss = 12 - len(str(image_id))
            image_unique_id = ("0" * miss) + str(str(image_id))
            image_ids.append(image_unique_id)
            categ_ids.append(json.dumps(js['annotations'][i]['category_id']))

        dataset = {'ImageID': image_ids,'Categories':categ_ids}
        dataset = pd.DataFrame.from_dict(dataset)
        dataset = dataset.groupby('ImageID', as_index=False).agg(lambda x: x.tolist())
        dataset
        print(len(dataset))
        self.landmarks_frame = dataset
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


def get_metrics():
    #Classifier metrics. (Download COCO 2017 dataset for that.) 
    #Here is the summary of the accuracy for the model trained on the instances set of COCO train2017 and evaluated on COCO   
    #val2017. Box AP = 37.0
    test_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                     ])
    coco_dataset = COCO2017(json_file='obj_detect_pytorch/obj_detect_pytorch/dataset/stuff_val2017.json',
                        root_dir='obj_detect_pytorch/obj_detect_pytorch/dataset/val2017/',
                        transform = test_transforms)
    loader = torch.utils.data.DataLoader(coco_dataset, batch_size=5)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    to_pil = transforms.ToPILImage()
    
    for ii in range(len(images)):
        image = to_pil(images[ii])
        index = predict_image(image)
        sub = fig.add_subplot(1, len(images), ii+1)
        res = int(labels[ii]) == index
       

    with open('categories', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(index)
        
    return "hi"
 

def predict(**args):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    """
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
    """
    a = get_metrics()
    
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
        #message = mutils.format_prediction(pred_boxes,pred_class, pred_score)  
        return a

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
