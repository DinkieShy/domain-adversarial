import json
import csv
import datetime
import os

def convertDataset(datasetFile = ""):
    INPUT_DIR = "./input/"

    assert datasetFile != "", "No dataset passed"
    assert os.path.exists(datasetFile), datasetFile + " does not exist"

    csvFile = open(datasetFile, newline='')
    data = list(csv.reader(csvFile))

    del data[0] #remove column titles

    csvFile.close()

    info = {
        "year": 2020,
        "version": "1",
        "description": "GWD dataset",
        "contributor": "placeholder",
        "url": "placeholder",
        "date_created": datetime.datetime.now()
    }

    licenseItem = {
        "id": 0,
        "name": "placeholder",
        "url": "placeholer"
    }

    categoryItem = {
        "id": 0,
        "name": "wheat",
        "supercategory": "plant"
    }

    cocoDataset = {
        "info": info,
        "images": [],
        "annotations": [],
        "categories": [categoryItem],
        "licenses": [licenseItem]
    }

    correctImageIDs = []

    for item in os.listdir(INPUT_DIR + "data/"):
        if item[-4:] == ".jpg":
            correctImageIDs.append(item[:-4])

    # data = [item for item in data if item[0] in correctImageIDs]

    imageIDs = {}
    lastImage = ""
    for i in range(len(data)):
        item = data[i]
        item[3] = json.loads(item[3])
        currentImage = item[0]
        if currentImage != lastImage:
            lastImage = currentImage
            imageItem = {
                "id": len(cocoDataset["images"]),
                "width": 1024,
                "height": 1024,
                "file_name": lastImage + ".jpg",
                "license": 0,
                "flickr_url": "placeholder",
                "coco_url": INPUT_DIR + "data/" + lastImage + ".jpg",
                "date_captured": datetime.datetime.now()
            }
            if currentImage in correctImageIDs:
                imageIDs[lastImage] = len(cocoDataset["images"])
                cocoDataset["images"].append(imageItem)

        annotationItem = {
            "id": i,
            "image_id": len(cocoDataset["images"])-1,
            "category_id": 0,
            "segmentation": [[
            int(item[3][0]), int(item[3][1]),
            int(item[3][0]+item[3][2]), int(item[3][1]),
            int(item[3][0]+item[3][2]), int(item[3][1]+item[3][3]),
            int(item[3][0]), int(item[3][1]+item[3][3])
            ]],
            "area": item[3][2]*item[3][3],
            "bbox": item[3],
            "iscrowd": 0
        }
        if currentImage in correctImageIDs:
            cocoDataset["annotations"].append(annotationItem)

    cocoDataSetFile = open(INPUT_DIR + "cocoDataset.json", "w")
    json.dump(cocoDataset, cocoDataSetFile, default=str)
    cocoDataSetFile.close()

    imageIDFile = open(INPUT_DIR + "imageIDs.json", "w")
    json.dump(imageIDs, imageIDFile)
    imageIDFile.close()