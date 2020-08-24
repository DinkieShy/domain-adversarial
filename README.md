# Domain Adversarial

This repository includes source code for adding a domain adversarial head to a pytorch network  
To run: add .csv files for training and validation sets in `input/`, then create `input/data/` and fill with images

This was created for the Global Wheat Detection 2020 challenge, https://www.kaggle.com/c/global-wheat-detection/overview

# Installation
Prerequisites:  
-Cuda 10.0, https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html  
-Python >= 3.6  
-Latest version of pip, `pip install --upgrade pip`

Install:  
Clone this repository, then run `pip install -r requirements.txt` in the project directory

OPTIONAL: You may wish to create a new virtual environment to install these packages, by running `python -m venv envName`, then updating using `pip install --upgrade pip`.  
To use this virtual environment, you must run `source envName/bin/activate` in any new terminal.  
(You can replace `envName` in both commands with whatever you want your environment to be called)

# Use
## Training
Create a config file in `configs/` [(an example is present)](./configs/config.txt), then run `python whileTrain.py configFile.txt`  

Training flags:  
`-q` | `--quiet`       - Reduce output  
`-s` | `--show_images` - Show images while training (only useful for verifying data augmentation)  
`-r` | `--resume`      - Resume training from a previous run

### Resuming
During training, an in-progress model is saved after every epoch. To resume training, use the `-r` flag at runtime (as above) with the config file of the regime you want to resume. Doing this will resume the most recent entry in `output/` with a matching config name (note: this is the name set *within* the file, *not* the name of the file).  
In order to resume training a different attempt, you can simply rename the folder containing the output you wish to resume from and make a new config file, or simply change the time to make it the most recent.

## Inference
Run `python inference.py`, passing paths to models (.pth files) and a test set  
Results will be saved in the same directories as the models

Inference flags:  
`-c`   | `--coco`        - Output results in COCO format, and create `cocoDataset.json`  
`-s x` | `--show x`      - Visualise a sample of detections, with x being the number of samples  
`-t x` | `--threshold x` - Don't include detections with a confidence score below this threshold

When showing images, the colour of the bounding box represents the confidence score.  
Red - less then 0.6  
Yellow - less than 0.9  
Green - 0.9 and above

# Evaluation
It is recommended that the COCO api is used for error evaluation https://github.com/wenmengzhou/cocoapi/tree/add_analyze_func  
In order to use analyze() (currently only provided in MATLAB, [based on work by Hoiem et al](http://dhoiem.cs.illinois.edu/projects/detectionAnalysis/)), replace `cocoapi/PythonAPI/pycocotools` with the version from [this PR](https://github.com/wenmengzhou/cocoapi/tree/add_analyze_func)  
Set -c flag when running the inference script to produce `./input/cocoDataset.json` which can be used for COCO evaluation, along with the cocoResults.json generated at the model path

# TODO:
Make graph once training is complete using trainingLog.txt  
CLI flag for switching between models
Focus on informative graphs, need to demonstrate performance delta between models
