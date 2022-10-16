import pandas as pd
# import numpy as np
from sklearn.linear_model import LinearRegression


def main():
    
    df = pd.read_csv('out.csv')
    x_colname = [x for x in df.columns if 'y' in x]
    print(x_colname)
    y_colname = ['x']

    THRESHOLD = 1 / len(x_colname)

    lr = LinearRegression()
    X = df[x_colname]
    y = df[y_colname]
    reg = lr.fit(X, y)
    print(f'score: {reg.score(X, y)}')
    print(f'coef: {reg.coef_}, {type(reg.coef_)}')
    drop_list = (abs(reg.coef_) < 0.007).reshape(-1)
    print(sum(abs(reg.coef_).reshape(-1)))
    print(drop_list.shape)

    X = X.loc[:, drop_list]
    lr2 = LinearRegression()
    reg = lr2.fit(X, y)
    print('iter 2')
    print(f'score: {reg.score(X, y)}')
    print(f'coef: {reg.coef_}, {type(reg.coef_)}')

if __name__ == '__main__':
    main()