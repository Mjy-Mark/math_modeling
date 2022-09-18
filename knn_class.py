import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN
from im_data import my_data

data = my_data('合并1.1.xlsx')
df = data.get_data()
df.loc[df['类型'] == '高钾', ['类型']] = 1
df.loc[df['类型'] == '铅钡', ['类型']] = 2
df_train = df.values[:, 5:20]
data_test = my_data('atch3.xlsx')
df_test = data_test.get_data()
df_test.loc[df_test['表面风化'] == '无风化', ['表面风化']] = 0
df_test.loc[df_test['表面风化'] == '风化', ['表面风化']] = 1
df_test = df_test.values[:, 1:]
df_label = df.values[:, 3]
# print(df_train)
# print(df_test)
# print(df_label)
knn = kNN(n_neighbors=3, weights='distance')
knn.fit(df_train, df_label.astype('int'))
pre_col = knn.predict_proba(df_test)
print(pre_col)