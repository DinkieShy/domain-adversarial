import matplotlib
import sys
import os
from util.ReadLogFile import readLogFile
import matplotlib.pyplot as plt
from datetime import datetime

configName = "default"
args = sys.argv
for i in range(len(args)):
    if args[i]:
        configName = args[i]

data = readLogFile(configName)

x  = []
totalLoss = []
domainLoss = []
precision = []

for i in range(len(data)):
    x.append(data[i]['iteration'])
    totalLoss.append(data[i]['totalLoss'])
    domainLoss.append(data[i]['domainLoss'])
    precision.append(None if data[i]['precision'] == -1 else data[i]['precision'])

fig, ax = plt.subplots()
ax.plot(x, totalLoss, label="total loss")
ax.plot(x, domainLoss, label="domain head loss")
ax.plot(x, precision, label="precision")

ax.set_xlabel("iteration")
ax.legend()

plt.show()

saveLocation = "./output/"
outputPaths = [path for path in os.listdir(saveLocation) if os.path.isdir(saveLocation + path) and configName in path]
outputPaths.sort(key=lambda x: datetime.strptime(x, configName + '_%Y %m %d_%H %M'), reverse=True)
assert len(outputPaths) > 0, "output for config " + configName + " does not exist"
saveLocation += outputPaths[0] + "/graph.png"

plt.savefig(saveLocation)