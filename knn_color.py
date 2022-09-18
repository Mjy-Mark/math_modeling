import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN
from im_data import my_data

data_train = my_data('knn_data.xlsx')
data_test = my_data('knn_test.xlsx')
df_train = data_train.get_data().values[:, 2:]
df_test = data_test.get_data().values[:, 1:]
df_label = np.array(data_train.get_data().values[:, 1])
# print(df_train)
# print(df_test)
# print(df_label)
knn = kNN(n_neighbors=2, weights='distance')
knn.fit(df_train, df_label.astype('int'))
pre_col = knn.predict_proba(df_test)
print(pre_col)
