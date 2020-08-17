# Contains scripts for calculating evaluation metrics

from .Boxes import BoundingBox, TargetBoundingBox
import numpy as np

def calculatePrecision(outputs, targets, image_ids, IOUTHRESHOLD, CONFIDENCE_THRESHOLD):
    #Outputs are in the form of a dict
    #'boxes'          : array of predicted bounding boxes
    #'scores'         : array of confidence scores
    # everything else : Kinda useless in terms of evaluating the accuracy

    formattedOutputs = {}

    for i in range(len(outputs)):
        newOutput = []
        for ii in range(len(outputs[i]['boxes'])):
            newBox = {"score": outputs[i]['scores'].cpu()[ii].item(), "box": outputs[i]['boxes'].cpu()[ii].detach().numpy().astype(np.int32)}
            if newBox["score"] >= CONFIDENCE_THRESHOLD:
                newOutput.append(newBox)
        formattedOutputs[image_ids[i]] = newOutput

    #New format: formattedOutputs[image_id][index of box](["score"] | ["box"])

    #Combine outputs and target array for easier evaluation

    combinedOutputs = {}

    for i in range(len(image_ids)):
        targetBoxes = targets[i]["boxes"].cpu().numpy().astype(np.int32)
        predictedBoxes = []
        predictionScores = []

        for ii in range(len(formattedOutputs[image_ids[i]])):
            predictedBoxes.append(formattedOutputs[image_ids[i]][ii]["box"])
            predictionScores.append(formattedOutputs[image_ids[i]][ii]["score"])

        newCombination = {"targetBoxes": targetBoxes, "predictedBoxes":predictedBoxes, "predictionScores":predictionScores, "imageSize": [1024, 1024]}
        combinedOutputs[image_ids[i]] = newCombination

    #Combined format: combinedOutputs[image id](["targetBoxes"] | ["predictedBoxes"] | ["predictionScores"] | ["imageSize"])[index of expected array]
    #Note: predictedBoxes and predictionScores will have the same length, but targetBoxes may not

    targetBoxes = []
    for box in combinedOutputs["targetBoxes"]:
        targetBoxes.append(TargetBoundingBox(box[0], box[1], box[2]-box[0], box[3]-box[1]))

    predictedBoxes = []
    for box in combinedOutputs["predictedBoxes"]:
        predictedBoxes.append(BoundingBox(box[0], box[1], box[2]-box[0], box[3]-box[1]))

    totalBoxes = len(predictedBoxes)
    truePositive = 0

    for targetBox in targetBoxes:
        for i in range(len(predictedBoxes)-1, -1, -1):
            fit = targetBox.intersect(predictedBoxes[i])
            if fit/targetBox.union(predictedBoxes[i]) >= IOUTHRESHOLD:
                truePositive += 1
                del predictedBoxes[i] #Remove predicted box from list if it hit a target box (can't hit more than one)

    # print(truePositive, "correct out of", totalBoxes, "predictions.")

    return truePositive / totalBoxes