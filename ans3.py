import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import xlrd as xr
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from im_data import my_data

df = my_data()


