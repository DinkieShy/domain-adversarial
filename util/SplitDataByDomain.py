import pandas as pd
import csv
import numpy as np
import math
from shutil import copyfile
import json

train_df = pd.read_csv("./input/fullTrain.csv")

domainsToRemove = ["usask_1", "inrae_1", "rres_1"]

# # Print details of domains for choosing domains to remove

# domains = []
# domainCounts = []
# for i in range(len(train_df)):
#     source = train_df['source'][i]
#     if source in domains:
#         domainCounts[domains.index(source)] += 1
#     else:
#         domains.append(source)
#         domainCounts.append(1)

# print(domains)
# print(domainCounts)
# print(sum(domainCounts))

train = pd.DataFrame(columns=train_df.columns)
test  = pd.DataFrame(columns=train_df.columns)

print("Splitting data...")

mask = train_df['source'].map(lambda source: source in domainsToRemove)

train = train_df[~mask]
test = train_df[mask]

print("Saving...")

train.to_csv("./input/trainDomain.csv", index=False)
test.to_csv("./input/testDomain.csv", index=False)

print("Done.")