from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('data1026.csv', header=None)
bws = np.linspace(31, 50, 10)
for bw in bws:
    bw = round(bw, 3)
    print(f"正在计算带宽为{bw}\n")
    ms = MeanShift(bandwidth=bw)
    ms.fit(data)
    fig = plt.figure()
    # 创建绘图区域
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[0], data[1], data[2], s=3, c=ms.labels_)

    plt.axis('off')
    plt.savefig(f'image/B{bw}.png', dpi=1000, bbox_inches='tight', pad_inches=0)
