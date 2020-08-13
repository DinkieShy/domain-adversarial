from os import path

class LearningRateCascade():
    def __init__(self):
        self.learningRate = -1
        self.epochsToRun = -1
        self.epochsUntilChange = -1
        self.minEpochs = 0
        self.performanceThreshold = 0.005

    def assertValid(self):
        assert self.learningRate != -1, "Learning rate must be set in config file"
        assert self.epochsToRun != -1 or self.epochsUntilChange != -1, "Either epochsToRun or epochsUntilChange must be set"
        return True

    def isValid(self):
        return self.learningRate != -1 and (self.epochsToRun != -1 or self.epochsUntilChange != -1)

    def __str__(self):
        return "learningRate: " + str(self.learningRate) + "\nepochsToRun: " + \
            str(self.epochsToRun) + "\nepochsUntilChange: " + str(self.epochsUntilChange) + \
            "\nminEpochs: " + str(self.minEpochs) + "\nperformanceThreshold: " + str(self.performanceThreshold)

def readConfigFile(filename, INPUT_DIR):
    configFile = open(filename)
    content = configFile.readlines()
    configFile.close()

    trainFile = ""
    validFile = ""
    iouThreshold = -1
    confidenceThreshold = -1
    learningRates = [LearningRateCascade()]
    name = ""

    for line in content:
        if line[0] != "#":
            splitLine = [string.strip().lower() for string in line.split("=")]

            if splitLine[0] == "learningrate":
                if learningRates[-1].isValid():
                    learningRates.append(LearningRateCascade())
                learningRates[-1].learningRate = float(splitLine[1])

            elif splitLine[0] == "epochstorun":
                learningRates[-1].epochsToRun = int(splitLine[1])

            elif splitLine[0] == "epochsuntilchange":
                learningRates[-1].epochsUntilChange = int(splitLine[1])

            elif splitLine[0] == "minepochs":
                learningRates[-1].minEpochs = int(splitLine[1])

            elif splitLine[0] == "performancethreshold":
                learningRates[-1].performanceThreshold = float(splitLine[1])

            elif splitLine[0] == "train":
                trainFile = splitLine[1]

            elif splitLine[0] == "valid":
                validFile = splitLine[1]

            elif splitLine[0] == "iouthreshold":
                iouThreshold = splitLine[1]

            elif splitLine[0] == "confidencethreshold":
                confidenceThreshold == splitLine[1]

            elif splitLine[0] == "name":
                name = splitLine[1]

    validating = False
    for i in learningRates:
        i.assertValid()
        if i.epochsToRun == -1:
            validating = True

    assert path.exists(INPUT_DIR + trainFile), "Train set does not exist"
    if validating:
        assert path.exists(INPUT_DIR + validFile), "Validation set does not exist"
    
    return trainFile, validFile, iouThreshold, confidenceThreshold, learningRates, name
    
if __name__ == "__main__":
    trainFile, validFile, iouThreshold, confidenceThreshold, learningRates, configName = readConfigFile("configs/config.txt", "input/")
    print(configName, "---------------------------", trainFile, validFile, iouThreshold, confidenceThreshold, sep="\n")
    for i in learningRates:
        print()
        print(i)
    print(len(learningRates))