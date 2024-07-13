import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
import numpy as np


data1 = pd.read_csv('data1030.csv', names=['x','y'])
bws = np.linspace(1, 10, 11)
for bw in bws:
    bw = round(bw, 3)
    ms = cluster.MeanShift(bandwidth=bw)
    ms.fit(data1)
    labels1 = pd.DataFrame(ms.labels_,columns=['Cluster ID'])
    result1 = pd.concat((data1,labels1), axis=1)
    result1.plot.scatter(x='x',y='y',c='Cluster ID',colormap='jet')
    plt.axis('off')
    plt.savefig(f'image/B{bw}.png', dpi=1000, bbox_inches='tight', pad_inches=0)