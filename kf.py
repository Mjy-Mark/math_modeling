from im_data import my_data
from scipy.stats import chi2_contingency
import numpy as np

color = my_data('color.xlsx')
cls = my_data('class.xlsx')
orn = my_data('orn.xlsx')
col_data = color.get_data().values[:, 1:]
cls_data = cls.get_data().values[:, 1:]
orn_data = orn.get_data().values[:, 1:]
kf_col = chi2_contingency(col_data)
kf_cls = chi2_contingency(cls_data)
kf_orn = chi2_contingency(orn_data)
print("颜色：", kf_col)
print("种类：", kf_cls)
print("纹饰：", kf_orn)
