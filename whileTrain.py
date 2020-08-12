import pandas as pd
import numpy as np
import os
import re
import cv2
import sys
import datetime

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

from util.Evaluation import calculatePrecision
from datasets.GWD import WheatDataset
from models.domainAdversarial import DomainAdversarialHead
from util.parseConfig import readConfigFile

INPUT_DIR = "./input/"
OUTPUT_DIR = "./output/"
IMAGE_DIR = INPUT_DIR + "data/"

REDUCED_OUTPUT = False
SHOW_IMAGES = False
configFile = "./configs/config.txt"

resume = False

args = sys.argv
for i in range(len(args)):
    if args[i] == "-q" or args[i] == "--quiet":
        REDUCED_OUTPUT = True
    elif args[i] == "-s" or args[i] == "--show-images":
        SHOW_IMAGES = True
    elif args[i] == "-r" or args[i] == "--resume":
        resume = True
    elif ".txt" in args[i]:
        configFile = "./configs/" + args[i]

assert os.path.exists(configFile), "Config file " + configFile + " does not exist"

# Training options: 
learningRatesToUse = []

trainFile, validFile, IOU_THRESHOLD, CONFIDENCE_THRESHOLD, learningRates, configName = readConfigFile(configFile, INPUT_DIR)
for i in learningRates:
    learningRatesToUse.append((i.learningRate, i.epochsToRun, i.epochsUntilChange, i.minEpochs, i.performanceThreshold))

currentTime = datetime.datetime.today()
currentTimeString = str(currentTime.year) + ":" + str(currentTime.month) + ":" + \
    str(currentTime.day) + "_" + str(currentTime.hour) + ":" + str(currentTime.minute)

OUTPUT_DIR += configName + "_" + currentTimeString + "/"

# Saving
model_path_base = OUTPUT_DIR + "checkpoints/lr-" # Saves best and final for each learning rate
IN_PROGRESS_PATH = OUTPUT_DIR + "checkpoints/trainingInProgess.pth.tar" # Path to save in-progress model
LOG_PATH = OUTPUT_DIR + "trainingLog.txt"

train_df = pd.read_csv(INPUT_DIR + trainFile) # CSV containing the training set
valid_df = pd.read_csv(INPUT_DIR + validFile) # CSV containing the validation set

directories = [INPUT_DIR, OUTPUT_DIR, IMAGE_DIR, OUTPUT_DIR + "checkpoints/"]

for directory in directories:
    if not os.path.exists(directory):
        os.mkdir(directory)

train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

valid_df['x'] = -1
valid_df['y'] = -1
valid_df['w'] = -1
valid_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

valid_df[['x', 'y', 'w', 'h']] = np.stack(valid_df['bbox'].apply(lambda x: expand_bbox(x)))
valid_df.drop(columns=['bbox'], inplace=True)
valid_df['x'] = valid_df['x'].astype(np.float)
valid_df['y'] = valid_df['y'].astype(np.float)
valid_df['w'] = valid_df['w'].astype(np.float)
valid_df['h'] = valid_df['h'].astype(np.float)

train_ids = train_df['image_id'].unique()
valid_ids = valid_df['image_id'].unique()

print("Training on:", len(train_ids))
print("Evaluating on:", len(valid_ids))

# Albumentations
def get_train_transform():
    return A.Compose([
        A.Flip(p=0.5),
        A.Rotate(20, p=0.9),
        # rescale aspect ratio - check if it looks realistic
        # A.RandomSizedCrop((975, 1020), 1024, 1024, (0.75, 1.33), p=0.8),
        A.RandomSizedBBoxSafeCrop(1024, 1024, 0.2),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

#--- Creating the model -----------------------------------------------------------------------------------------------------

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# model.roi_heads.box_predictor.add_module("DomainAdversarial", DomainAdversarialHead())

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = WheatDataset(train_df, IMAGE_DIR, get_train_transform())
valid_dataset = WheatDataset(valid_df, IMAGE_DIR, get_valid_transform())

train_data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE!!!")
else:
    print("Cuda available!")

#--- Train

model.to(device)

loss_hist = Averager()

def evaluate(valid_data_loader, model):
    with torch.no_grad():
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.eval().to(device)
        for images, targets, image_ids in valid_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

            IoUScore = 0
            imagesChecked = 0

            newIoUScore = calculatePrecision(outputs, targets, image_ids, IOU_THRESHOLD, CONFIDENCE_THRESHOLD)
            imagesChecked += len(image_ids)
            # print(newIoUScore)
            IoUScore += newIoUScore

            # print("Mean IoU so far:", IoUScore/imagesChecked)

    return IoUScore/imagesChecked

def loadCheckpoint(model, optimizer):
    saved = torch.load(IN_PROGRESS_PATH)
    model.load_state_dict(saved['state_dict'])
    optimizer.load_state_dict(saved['optimizer'])
    return model, optimizer, saved['epoch'], saved['outputPath']

def saveLogData(iterationCount, loss):
    logFile = open(LOG_PATH, "a")
    logFile.write(",\n" + str([iterationCount, loss]))
    logFile.close()

epochCount = 0
iterationCount = 0
bestPrecision = 0
lastChanged = 0
loadedSave = False

for learningRate, timeToRun, epochsUntilChange, minEpochs, performanceThreshold in learningRatesToUse:
    previousPrecisionValues = []
    DONE = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learningRate, momentum=0.9, weight_decay=0.0005)

    if resume:
      if not loadedSave:
        model, optimizer, epochCount, OUTPUT_DIR = loadCheckpoint(model, optimizer)
        iterationCount = epochCount * 3373
        loadedSave = True
        print("Loaded previous model state")

        model_path_base = OUTPUT_DIR + "checkpoints/lr-"
        IN_PROGRESS_PATH = OUTPUT_DIR + "checkpoints/trainingInProgess.pth.tar"
        LOG_PATH = OUTPUT_DIR + "trainingLog.txt"

      if epochCount > timeToRun + lastChanged and timeToRun != -1:
        lastChanged += timeToRun
        continue
      resume = False

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    while not DONE:
        model.train()
        # model.to(torch.double)
        loss_hist.reset()
        print("")
        for images, targets, image_ids in train_data_loader:
            # images = list(image.to(device).to(torch.double) for image in images)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if SHOW_IMAGES:
                imageToShow = images[0]
                cv2.imshow("image", imageToShow.to(torch.device('cpu')).permute(1, 2, 0).numpy())
                cv2.waitKey(1)

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            iterationCount += 1

            if not REDUCED_OUTPUT:
                progress = int(((iterationCount%(3373))/(3373))*50)
                print("\rProgress: [", "="*progress, ">", " "*(49-progress), "] ", iterationCount, end="", sep="")

        print("\nIterations:", str(iterationCount))
        saveLogData(iterationCount, loss_hist.value)
        
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print("Epoch #" + str(epochCount) + " loss: " + str(loss_hist.value))
        epochCount += 1

        if epochCount >= minEpochs and timeToRun == -1:
            # model.to(torch.float)
            precision = evaluate(valid_data_loader, model)

            previousPrecisionValues.append(precision)
            if len(previousPrecisionValues) > epochsUntilChange:
                del previousPrecisionValues[0]
                if all(abs(precision - previousPrecisionValue) < performanceThreshold for previousPrecisionValue in previousPrecisionValues):
                    DONE = True
            if precision >= bestPrecision:
                bestPrecision = precision
                torch.save(model.state_dict(), model_path_base + str(learningRatesToUse.index((learningRate, timeToRun))) + "-best.pth")

            print("Precision:", precision)
        elif epochCount >= timeToRun+lastChanged and timeToRun != -1:
            DONE = True
            print("Fixed epoch count reached")

        state = {'epoch': epochCount, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'outputPath': OUTPUT_DIR}
        torch.save(state, IN_PROGRESS_PATH)
    print("\nDone with learning rate:", learningRate, "\n")
    torch.save(model.state_dict(), model_path_base + str(learningRatesToUse.index((learningRate, timeToRun))) + "-final.pth")
    lastChanged = epochCount

print("\nCompleted")
