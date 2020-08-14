import pandas as pd
import numpy as np
import os
import re
import sys
import cv2

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from models.fasterRCNNWrapper import customFasterRCNN
from util.DatasetToCoco import convertDataset

import json

IMG_DIR = './input/data/'

WEIGHTS_FILES = []
testFile = ""
COCO = False

args = sys.argv
for i in range(len(args)):
    if ".pth" in args[i]:
        WEIGHTS_FILES.append(args[i])
    elif ".csv" in args[i]:
        testFile = args[i]
    elif args[i] == "-c" or args[i] == "--coco":
        COCO=True

assert len(WEIGHTS_FILES) > 0, "Model not selected, be sure to pass a weights file"
assert testFile != "", "Test set not selected, be sure to pass a csv file containing a test set"

if COCO:
    print("Converting dataset to COCO format...")
    convertDataset(testFile)

imageFiles = []

imageIDs = pd.read_csv("./test.csv")['image_id'].unique()

for imageID in imageIDs:
    imageFiles.append(IMG_DIR + imageID)

test_df = pd.DataFrame([[imageIDs[0],""]], columns=["image_id", "PredictionString"])

test_df = pd.concat([pd.DataFrame([[imageID, ""]], columns=["image_id", "PredictionString"]) for imageID in imageIDs[1:]], ignore_index=True)

class WheatTestDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

# Albumentations
def get_test_transform():
    return A.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE!!!")
else:
    print("Cuda available!")

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = customFasterRCNN(in_features, num_classes)

def collate_fn(batch):
    return tuple(zip(*batch))

test_dataset = WheatTestDataset(test_df, IMG_DIR, get_test_transform())

test_data_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    drop_last=False,
    collate_fn=collate_fn
)

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)

def format_bbox_string(boxes):
    strings = []
    for box in boxes:
        strings.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
    return strings

imageIDFile = open("./output/imageIDs.json")
imageIDConvert = dict(json.load(imageIDFile))
imageIDFile.close()

print(len(imageIDConvert))

for weights in WEIGHTS_FILES:
    model.load_state_dict(torch.load(weights))

    x = model.to(device)
    model.eval()

    detection_threshold = 0
    results = []

    for images, image_ids in test_data_loader:

        images = list(image.to(device) for image in images)
        outputs = model(images)

        for i, image in enumerate(images):

            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()
            
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            scores = [float(score) for score in scores]
            image_id = image_ids[i]
            
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            
            result = {
                'image_id' : image_id,
                'category_id' : 0,
                'bbox' : format_bbox_string(boxes),
                'score' : scores
            }
            
            results.append(result)

        if COCO:
            cocoResults = []

            for result in results:
                for i in range(len(result["bbox"])):
                    cocoResults.append({
                        'image_id' : imageIDConvert[result['image_id']],
                        'category_id' : 0,
                        'bbox': result['bbox'][i],
                        'score': result['score'][i]
                    })

            cocoResultsFile = open("./output/cocoResults-" + str(WEIGHTS_FILES.index(weights)) + ".json", 'w')
            cocoResultsFile.write(json.dumps(cocoResults))
            cocoResultsFile.close()
        else:
            resultsFile = open("./output/detections.json", "w")
            resultsFile.write(json.dumps(results))
            resultsFile.close()

    print("Done!")