import numpy as np
import pandas as pd
from im_data import my_data
from sklearn.decomposition import PCA

df = my_data('合并1.1.xlsx')
compon = df.get_data().values[1:, 5:19]
pca = PCA(n_components=2)
pca.fit(compon)
print(pca.explained_variance_ratio_)
print(pca.transform(compon))
