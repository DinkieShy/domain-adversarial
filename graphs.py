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
domainLossFound = True

lines = []
line = {'x': [], 'totalLoss': [], 'domainLoss': [], 'precision': []}

for i in range(len(data)):
    if 'iteration' in data[i]:
        line['x'].append(data[i]['iteration'])
        line['totalLoss'].append(data[i]['totalLoss'])
        line['precision'].append(None if data[i]['precision'] == -1 else data[i]['precision'])
        if domainLossFound and 'domainLoss' in data[i]:
            line['domainLoss'].append(data[i]['domainLoss'])
        else:
            domainLossFound = False
    else:
        lines.append(line)
        line = {'x': [], 'totalLoss': [], 'domainLoss': [], 'precision': []}

lines.append(line)

plt.style.use("seaborn")
fig, ax = plt.subplots()
for i in range(len(lines)):
    ax.plot(lines[i]['x'], lines[i]['totalLoss'], label="total loss" if i == 0 else None, color="r")
    if len(line['domainLoss']) > 0:
        ax.plot(lines[i]['x'], lines[i]['domainLoss'], label="domain head loss" if i == 0 else None, color="g")
    if not all(ii is None for ii in lines[i]['precision']):
        ax.plot(lines[i]['x'], lines[i]['precision'], label="precision" if i == 0 else None, color="b")

ax.set_xlabel("iteration")
ax.legend()

plt.show()

saveLocation = "./output/"
outputPaths = [path for path in os.listdir(saveLocation) if os.path.isdir(saveLocation + path) and path.split('_')[0] == configName]
outputPaths.sort(key=lambda x: datetime.strptime(x, configName + '_%Y %m %d_%H %M'), reverse=True)
assert len(outputPaths) > 0, "output for config " + configName + " does not exist"
saveLocation += outputPaths[0] + "/graph.png"

plt.savefig(saveLocation)