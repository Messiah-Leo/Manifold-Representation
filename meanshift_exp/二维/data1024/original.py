from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('data1024.csv', header=None)
bws = np.linspace(5, 35, 30)
for bw in bws:
    bw = round(bw, 3)
    print(f"正在计算带宽为{bw}\n")
    ms = MeanShift(bandwidth=bw)
    ms.fit(data)
    plt.scatter(data[0], data[1], s=3, c=ms.labels_)

    plt.axis('off')
    plt.savefig(f'image/B{bw}.png', dpi=1000, bbox_inches='tight', pad_inches=0)
