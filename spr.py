from im_data import my_data
import pandas as pd

data = my_data('spearman.xlsx')
df = data.get_data()
print(df.corr(method='spearman'))
