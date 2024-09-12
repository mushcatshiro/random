import re
from typing import Any, Union

import pandas as pd
import numpy as np
from scipy import stats


pattern = re.compile(r"mp[0-9]s")

class DataGenerator:
    def __init__(
        self, fname, ycol, datecol, groupcol, inp_type, outlier_mode, precision=3,
        norm_alpha=1e-3, fillup=False
    ):
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
        self.precision = precision
        self.outlier_mode = outlier_mode
        self.norm_alpha = norm_alpha
        self.fillup = fillup
        self.metadata = {"dataset": {}}
        self.generate_summary()
        print(self.metadata)

    def generate_summary(self):
        null_count = self._dataset.isnull().values.sum()
        full_count = self._dataset.size
        sparsity = round(null_count / full_count, self.precision)
        self.metadata["dataset"]["sparsity"] = sparsity
        print(f"sparsity: {round(sparsity * 100, 2)}%")

        for col in self.cols + [self.ycol]:
            print(f"processing {col}")

            # sparsity
            null_count = self._dataset[col].isnull().values.sum()
            full_count = self._dataset[col].size
            data_points = full_count - null_count
            sparsity = round(null_count / full_count, self.precision)
            print(null_count)

            tmp: np.ndarray = self._dataset[col].to_numpy()

            # basic properties
            if null_count == 0:
                mean = round(tmp.mean(), self.precision)
                if data_points < 30:
                    std = round(tmp.std(ddof=1), self.precision)
                else:
                    std = round(tmp.std(), self.precision)
                range = round(
                    tmp.max() - tmp.min(),
                    self.precision
                )
            else:
                mean = round(np.nanmean(tmp), self.precision)
                print(mean)
                if data_points < 30:
                    std = round(np.nanstd(tmp, ddof=1), self.precision)
                else:
                    std = round(np.nanstd(tmp), self.precision)
                range = round(
                    np.nanmax(tmp) - np.nanmin(tmp),
                    self.precision
                )

            # outlier
            outlier_summary = None
            if pattern.findall(self.outlier_mode):
                n = int(self.outlier_mode[2])
                gt_count = np.sum(tmp > (mean + n * std))
                lt_count = np.sum(tmp < (mean - n * std))
                outlier_summary = {
                    "gt_count": gt_count,
                    "lt_count": lt_count
                }
            elif self.outlier_mode.lower() == "iqr":
                q3 = np.quantile(tmp, 0.75)
                q2 = np.quantile(tmp, 0.5)
                q1 = np.quantile(tmp, 0.25)
                iqr = q3 - q1
                gt_count = np.sum(tmp > (q2 + 1.5 * iqr))
                lt_count = np.sum(tmp < (q2 + 1.5 * iqr))
                outlier_summary = {
                    "gt_count": gt_count,
                    "lt_count": lt_count
                }

            # normality check
            if np.sum(~np.isnan(tmp)) < 10:
                print("col has less than 10 samples")
                continue
            k, p = stats.normaltest(tmp, nan_policy="omit")        
            normality = False if p < self.norm_alpha else True
            print(
                f"k: {k}; p: {p}, "
                f"conclusion {'is' if normality else 'non'} normal"
            )

            self.metadata[col] = {}
            self.metadata[col]["sparsity"] = sparsity
            self.metadata[col]["size"] = full_count
            self.metadata[col]["mean"] = mean
            self.metadata[col]["std"] = std
            self.metadata[col]["range"] = range
            self.metadata[col]["outlier"] = outlier_summary
            self.metadata[col]["normality"] = normality

            sparsity = None
            null_count = None
            full_count = None
            data_points = None
            mean = None
            std = None
            range = None
            outlier_summary = None
            gt_count = None
            lt_count = None
            q1 = None
            q2 = None
            q3 = None
            iqr = None
            k = None
            p = None
            normality = None

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