import numpy as np
from numpy import linalg
import pandas as pd
from im_data import my_data
from sklearn.decomposition import PCA


def Pca(dataset, threshold=0.85):
    # 返回满足要求的特征向量
    ret = []
    data = []
    # 标准化
    for (index, line) in enumerate(dataset):
        dataset[index] -= np.mean(line)
        # np.std(line, ddof = 1)即样本标准差(分母为n - 1)
        # dataset[index] /= np.std(line, ddof=1)
    # 求协方差矩阵
    Cov = np.cov(dataset)
    # 求特征值和特征向量
    eigs, vectors = linalg.eig(Cov)
    # 第i个特征向量是第i列，为了便于观察将其转置一下
    for i in range(len(eigs)):
        data.append((eigs[i], vectors[:, i].T))
    # 按照特征值从大到小排序
    data.sort(key=lambda x: x[0], reverse=True)
    sum = 0
    for comp in data:
        sum += comp[0] / np.sum(eigs)
        ret.append(
            tuple(map(
                # 保留5位小数
                lambda x: np.round(x, 5),
                # 特征向量、方差贡献率、累计方差贡献率
                (comp[1], comp[0] / np.sum(eigs), sum)
            ))
        )
        print('特征值:', comp[0], '特征向量:', ret[-1][0], '方差贡献率:', ret[-1][1], '累计方差贡献率:', ret[-1][2])
        # if sum > threshold:
        #     return ret

print("总的")
df = my_data('合并1.1.xlsx')
compon = df.get_data().values
compon = np.matrix(compon, dtype='float64').T
Pca(compon)
print("高钾")
df_k = my_data('k.xlsx')
compon_k = df_k.get_data().values
compon_k = np.matrix(compon_k, dtype='float64').T
Pca(compon_k)
print("铅钡")
df_ba = my_data('ba.xlsx')
compon_ba = df_ba.get_data().values
compon_ba = np.matrix(compon_ba, dtype='float64').T
Pca(compon_ba)

# ingre = np.array(df.get_data().values[1:, 5:19])
# pca = PCA(n_components=2)
# pca.fit(ingre)
# print(pca.explained_variance_ratio_)
# print(pca.transform(ingre))
