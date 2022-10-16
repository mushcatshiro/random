import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from sklearn.datasets import make_spd_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)


def merge_join(data, id='x'):
    if len(data) > 1:
        mid = len(data) // 2
        left = data[:mid]
        right = data[mid:]

        # Recursive call on each half
        return merge_join(left, id).merge(merge_join(right, id), on=id, how='outer').dropna()

    if len(data) == 1:
        return data[0]

def main():
    ol = []
    for it in tqdm(range(0, 10)):
        size = 100000
        results = []
        for i in range(0, 100):
            l = np.random.randint(2, 20)
            mean = np.random.rand(l)
            cov = make_spd_matrix(l, random_state=0)

            pts = np.random.multivariate_normal(mean, cov, size=size)
            x, y = pts[:, 0], pts[:, 1]
            y = y + np.random.normal(0, 1, size)
            scaler = MinMaxScaler()
            x = scaler.fit_transform(x.reshape(-1, 1)).T[0]
            x = np.around(x, 5)
            t_df = pd.DataFrame(list(zip(x, y)), columns=['x', 'y' + str(i)])
            t_df.drop_duplicates('x', inplace=True)
            results.append(t_df)

        df = merge_join(results)
        df.sort_values('x', inplace=True)
        scaler = MinMaxScaler()
        df['x'] = scaler.fit_transform(df['x'].values.reshape(-1, 1)).T[0]

        x_colname = [x for x in df.columns if 'y' in x]
        y_colname = ['x']
        corr_matrix = df[x_colname].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='YlGnBu', vmax=1, vmin=-1)
        plt.show()
        lr = LinearRegression()
        reg = lr.fit(df[x_colname], df[y_colname])
        print(f'score: {reg.score(df[x_colname], df[y_colname])}')
        print(f'coef: {reg.coef_}')
        # print(np.amax(reg.coef_))
        # ol.append(max(reg.coef_).reshape(-1))
    # print(ol)


if __name__ == '__main__':
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument()
    # args = parser.parse_args()
    import datetime as dt
    start = dt.datetime.now()
    main()
    end = dt.datetime.now()
    print(f'{(end-start).total_seconds()}s')