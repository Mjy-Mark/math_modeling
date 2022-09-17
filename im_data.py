import pandas as pd


class my_data(object):
    """
    导入数据类
    """

    def __init__(self, name):
        self._df = pd.DataFrame(pd.read_excel(name))

    def get_data(self):
        return self._df


if __name__ == '__main__':
    data = my_data('合并1.1.xlsx')
    df = data.get_data()
    d = df.values[1:, 5:19]
    print(d)
