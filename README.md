# Domain Adversarial

This repository includes source code for adding a domain adversarial head to a pytorch network  
To run: add .csv files for training and validation sets in `input/`, then create `input/data/` and fill with images

This was created for the Global Wheat Detection 2020 challenge, https://www.kaggle.com/c/global-wheat-detection/overview

# Installation
Prerequisites:  
-Cuda 10.0, https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html  
-Latest version of pip, `pip install --upgrade pip`

Install:  
Clone this repository, then run `pip3 install -f requirements.txt` in the project directory

OPTIONAL: You may wish to create a new virtual environment to install these packages, by running `python3 -m venv envName`.  
To use this virtual environment, you must run `source envName/bin/activate` in any new terminal (not necessary immediately after creating the environment).  
(You can replace `envName` in both commands with whatever you want your environment to be called)
