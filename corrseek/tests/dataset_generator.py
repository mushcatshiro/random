import numpy as np
import pandas as pd


def dataset_generator(perc=10):
    r, c = 100, 10
    size = r * c
    mask = int(size * perc / 100)
    df1 = np.random.normal(size=(100, 10))
    df2 = np.copy(df1)
    df1.ravel()[np.random.choice(df1.size, mask, replace=False)] = np.nan
    df1 = pd.DataFrame(df1, columns=[f"col_{i}" for i in range(df1.shape[1])])
    df1["group"] = np.random.choice(["g1", "g2", "g3"], size=len(df1))
    df1["datetime"] = np.random.choice(["0000", "0001", "0002"], size=len(df1))
    return df1, df2

def fast_skewed_dataset_generator(N, alpha=0.0, loc=0.0, scale=1.0):
    sigma = alpha / np.sqrt(1.0 + alpha**2) 
    u0: np.ndarray = np.random.randn(N)
    v: np.ndarray = np.random.randn(N)
    u1 = (sigma*u0 + np.sqrt(1.0 - sigma**2)*v) * scale
    u1[u0 < 0] *= -1
    u1 = u1 + loc
    y = np.random.normal(size=(N, ))
    df1 = pd.DataFrame({"col_0": y, "col_1": u1})
    df1["group"] = np.random.choice(["g1", "g2"], size=len(df1))
    return df1