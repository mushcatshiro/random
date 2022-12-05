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
        self.metadata = {"dataset": {}}
        self.check_normality()
        self.check_overall_sparsity()
    
    def check_overall_sparsity(self):
        null_count = self._dataset.isnull().values.sum()
        full_count = self._dataset.size
        sparsity = null_count / full_count
        self.metadata["dataset"]["sparsity"] = sparsity
        print(f"sparsity: {round(sparsity * 100, 2)}%")

    @property
    def groups(self):
        return self._groups

    @property
    def cols(self):
        return self._cols

    def save(self):
        self._dataset.to_csv("ydrop.csv", index=False)

    def _read_data(self, fname, inp_type):
        """
        TODO dealing with total empty columns
        """
        if inp_type == "csv":
            df = pd.read_csv(fname)
        if inp_type == "nparray":
            df = fname

        # remove all nan from y
        print(df.shape)
        idx = np.where(df[self.ycol].notna())
        df = df.iloc[idx[0], :]
        print(df.shape)
        idx = np.where(df[self.groupcol].notna())
        df = df.iloc[idx[0], :]
        print(df.shape)
        return df

    def check_normality(self, alpha=1e-3):
        # TODO post check to generate corresponding cfg
        for col in self._cols:
            tmp = self._dataset[col].to_numpy() 
            # print(col)
            # print(tmp)
            if np.sum(~np.isnan(tmp)) < 10:
                print("col has less than 10 samples")
                continue
            k, p = stats.normaltest(tmp, nan_policy="omit")        
            conclusion = "non normal" if p < alpha else "normal"
            print(f"k: {k}; p: {p}, conclusion: {conclusion}")
        

    def generate(self, col, date=False, group=False) -> Union[np.ndarray, np.ndarray, Any]:
        """
        TODO
        return dataframe instead fpr group support
        """
        idx = np.where(self._dataset[col].notna())
        tmp = self._dataset.iloc[idx[0], :]
        if date:
            return tmp[self.ycol].to_numpy().reshape(-1, 1), tmp[col].to_numpy().reshape(-1, 1), tmp[self.datecol].to_numpy().reshape(-1, 1)
        return tmp[self.ycol].to_numpy().reshape(-1, 1), tmp[col].to_numpy().reshape(-1, 1), None