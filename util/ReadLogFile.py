import os
import datetime
import torch
import json

punctuationToRemove = ["'", "{", "}", "tensor(", " "]
intFields = ["iteration"]
floatFields = ["domainLoss", "totalLoss", "precision"]
boolFields = ["changedLR"]

def readLogFile(configName):
    LOG_FILE = "./output/"

    outputPaths = [path for path in os.listdir(LOG_FILE) if os.path.isdir(LOG_FILE + path) and configName in path]
    outputPaths.sort(key=lambda x: datetime.datetime.strptime(x, configName + '_%Y %m %d_%H %M'), reverse=True)
    assert len(outputPaths) > 0, "output for config " + configName + " does not exist"
    LOG_FILE += outputPaths[0] + "/trainingLog.txt"


    logDataFile = open(LOG_FILE)
    logData = logDataFile.readlines()
    logDataFile.close()

    del logData[0]

    logDict = []

    for i in range(len(logData)):
        newDictEntry = {}
        for pair in logData[i].split(','):
            if ":" in pair:
                for punctuation in punctuationToRemove:
                    pair = pair.replace(punctuation, "")
                key = pair.split(":")[0]
                value = pair.split(":")[1]

            if key in intFields:
                newDictEntry[key] = int(value)
            elif key in floatFields:
                newDictEntry[key] = float(value)
            elif key in boolFields:
                newDictEntry[key] = value == "True"
        if newDictEntry != {}:
            logDict.append(newDictEntry)

    return logDict

if __name__ == "__main__":
    readLogFile("default")