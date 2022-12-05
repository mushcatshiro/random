import pandas as pd
import numpy as np
from typing import Any, Union
from scipy import stats


class DataGenerator:
    def __init__(self, fname, ycol, datecol, groupcol, inp_type):
        """
        dealing with nan values
        - test normality and backfill with mean or median
        """
        self.ycol = ycol
        self.groupcol = groupcol
        self.datecol = datecol
        self._dataset: pd.DataFrame = self._read_data(fname, inp_type)
        self._groups = self._dataset[groupcol].unique()
        self._cols = [
            x for x in self._dataset.columns.tolist()
            if x not in [ycol, datecol, groupcol]
        ]
        self.metadata = {}
        self.check_normality()

    @property
    def groups(self):
        return self._groups

    @property
    def cols(self):
        return self._cols

    def save(self):
        self._dataset.to_csv("ydrop.csv", index=False)

    def _read_data(self, fname, inp_type):
        if inp_type == "csv":
            df = pd.read_csv(fname)

        # remove all nan from y
        print(df.shape)
        idx = np.where(df[self.ycol].notna())
        df = df.iloc[idx[0], :]
        print(df.shape)
        idx = np.where(df[self.groupcol].notna())
        df = df.iloc[idx[0], :]
        print(df.shape)
        return df

    def check_normality(self):
        for col in self._cols:
            tmp = self._dataset[col].to_numpy() 
            print(col)
            print(tmp)
            if np.sum(~np.isnan(tmp)) < 10:
                print("col has less than 10 samples")
                continue
            k, p = stats.normaltest(tmp, nan_policy="omit")        
            print(k, p)
        

    def generate(self, col, date=False) -> Union[np.ndarray, np.ndarray, Any]:
        idx = np.where(self._dataset[col].notna())
        tmp = self._dataset.iloc[idx[0], :]
        if date:
            return tmp[self.ycol].to_numpy().reshape(-1, 1), tmp[col].to_numpy().reshape(-1, 1), tmp[self.datecol].to_numpy().reshape(-1, 1)
        return tmp[self.ycol].to_numpy().reshape(-1, 1), tmp[col].to_numpy().reshape(-1, 1), None