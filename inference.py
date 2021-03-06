import pandas as pd
import numpy as np
import os
import re
import sys
import cv2
import random

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

from models.NotGeneralisedRCNN import DomainAwareRCNN
from util.DatasetToCoco import convertDataset

import json

IMG_DIR = './input/data/'
INPUT_DIR = "./input/"

WEIGHTS_FILES = []
testFile = ""
COCO = False
SHOW_IMAGES = False
IMAGES_TO_SHOW = 0
DETECTION_THRESHOLD = 0
USE_DOMAIN = True

args = sys.argv
for i in range(len(args)):
    if ".pth" in args[i]:
        WEIGHTS_FILES.append(args[i])
    elif ".csv" in args[i]:
        testFile = INPUT_DIR + args[i]
    elif args[i] == "-c" or args[i] == "--coco":
        COCO=True
    elif args[i] == "-s" or args[i] == "--show":
        SHOW_IMAGES = True
        IMAGES_TO_SHOW = int(args[i+1])
    elif args[i] == "-t" or args[i] == "--threshold":
        DETECTION_THRESHOLD = int(args[i+1])
    elif args[i] == "-n" or args[i] == "--nodomain":
        USE_DOMAIN = False

assert len(WEIGHTS_FILES) > 0, "Model not selected, be sure to pass a weights file"
assert testFile != "", "Test set not selected, be sure to pass a csv file containing a test set"

OUTPUT_DIRS = [weightsFile[:-4] + "/" for weightsFile in WEIGHTS_FILES]

for outputDir in OUTPUT_DIRS:
    assert os.path.exists(outputDir[:-1] + ".pth") and os.path.isfile(outputDir[:-1] + ".pth"), "Model file " + outputDir[:-1] + ".pth does not exist!"
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

# if SHOW_IMAGES and not os.path.exists("./output/results/

if COCO:
    print("Converting dataset to COCO format...")
    convertDataset(testFile) #doesn't return anything, but creates files necessary later for the COCO api

imageFiles = []

imageIDs = pd.read_csv(testFile)['image_id'].unique()

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

        image = cv2.imread(self.image_dir + "/" + image_id + ".jpg", cv2.IMREAD_COLOR)
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

if USE_DOMAIN:
    model = DomainAwareRCNN(num_classes=2, num_domains=10)
else:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE!!!")
else:
    print("Cuda available!")

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

for weights in WEIGHTS_FILES:
    try:
        model.load_state_dict(torch.load(weights))
    except RuntimeError:
        print("ERROR LOADING MODEL: Did you mean to use the --nodomain flag?")
        exit()

    x = model.to(device)
    model.eval()

    results = []

    imagesLeft = len(test_data_loader)
    imagesSaved = 0
    imageChance = float(IMAGES_TO_SHOW/imagesLeft)

    for images, image_ids in test_data_loader:

        images = list(image.to(device) for image in images)
        outputs = model(images)

        for i, image in enumerate(images):

            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()
            
            boxes = boxes[scores >= DETECTION_THRESHOLD].astype(np.int32)
            scores = scores[scores >= DETECTION_THRESHOLD]
            scores = [float(score) for score in scores]
            image_id = image_ids[i]

            if SHOW_IMAGES and imagesSaved <= IMAGES_TO_SHOW:
                if IMAGES_TO_SHOW-imagesSaved == imagesLeft or random.random() > imageChance:
                    imageToSave = image.permute(1,2,0).cpu().numpy()
                    for ii in range(len(boxes)):
                        start = (boxes[ii][0], boxes[ii][1])
                        stop = (boxes[ii][2], boxes[ii][3])
                        if scores[ii] < 0.6:
                            colour = (0, 0, 255)
                        elif scores[ii] < 0.9:
                            colour = (0, 255, 255)
                        else:
                            colour = (0, 255, 0)
                        cv2.rectangle(imageToSave, start, stop, colour, 2)
                    cv2.imwrite(OUTPUT_DIRS[WEIGHTS_FILES.index(weights)] + image_id + ".jpg", imageToSave*255)
                    imagesSaved += 1

            imagesLeft -= 1
            
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
            imageIDFile = open("./input/imageIDs.json")
            imageIDConvert = dict(json.load(imageIDFile))
            imageIDFile.close()

            cocoResults = []

            for result in results:
                for i in range(len(result["bbox"])):
                    cocoResults.append({
                        'image_id' : imageIDConvert[result['image_id']],
                        'category_id' : 0,
                        'bbox': result['bbox'][i],
                        'score': result['score'][i]
                    })

            cocoResultsFile = open(OUTPUT_DIRS[WEIGHTS_FILES.index(weights)] + "cocoResults.json", 'w')
            cocoResultsFile.write(json.dumps(cocoResults))
            cocoResultsFile.close()
        else:
            resultsFile = open(OUTPUT_DIRS[WEIGHTS_FILES.index(weights)] + "detections.json", "w")
            resultsFile.write(json.dumps(results))
            resultsFile.close()
    print("Results saved to", OUTPUT_DIRS[WEIGHTS_FILES.index(weights)])

    print("Done!")