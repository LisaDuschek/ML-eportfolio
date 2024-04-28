import numpy as np
import pandas as pd
import sklearn.cluster
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib.cm as cm
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("AB_NYC_2019.csv")

prices = df["price"].to_numpy()
nhbs = df["neighbourhood"].to_numpy()

df2 = df[["price", "neighbourhood"]].to_numpy() #selecting parameters of interest: modular
"""
kproto = KPrototypes(n_clusters = 5, init = 'Cao', n_jobs = -1)
clusters = kproto.fit_predict(df2, categorical = 1)
pd.series(clusters).value_counts()
"""
costs = []
n_clusters = []
clusters_assigned = []

for i in range(1,10): #running for optimal k
    try:
        kproto = KPrototypes(n_clusters = i, init = 'Cao', verbose = 2)
        clusters = kproto.fit_predict(df2, categorical = 1)
        costs.append(kproto.cost_)
        n_clusters.append(i)
        clusters_assigned.append(clusters)
    except:
        print("Invalid with {i} clusters")

#optimal k determined through elbow method
fig, ax = plt.subplots()
clt = ax.plot(n_clusters, costs, marker = 'o')
plt.show()
