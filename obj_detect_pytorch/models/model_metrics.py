import pandas as pd
import csv
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import glob
import torch
import os
import numpy as np

def predict_image(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    test_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                     ])
    image_tensor = image.float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output[0]['labels'].numpy()
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
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name)
        image = image.convert('RGB')
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
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                     ])
    coco_dataset = COCO2017(json_file='obj_detect_pytorch/obj_detect_pytorch/dataset/stuff_val2017.json',
                        root_dir='obj_detect_pytorch/obj_detect_pytorch/dataset/val2017/',
                        transform = test_transforms)

    index = []
    for i in range(5):  #Change to len(coco_dataset)
        sample = coco_dataset[i]
        image = sample['image']
        result_pred = predict_image(image)
        index.append(result_pred)
       
    with open('categories', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(index)
        
    return "Done"