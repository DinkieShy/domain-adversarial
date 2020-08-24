import matplotlib
import sys
import os
from util.ReadLogFile import readLogFile
import matplotlib.pyplot as plt

configName = "default"
args = sys.argv
for i in range(len(args)):
    if ".txt" in args[i]:
        configName = "./configs/" + args[i]

data = readLogFile(configName)

print(data)

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