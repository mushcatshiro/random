import unittest

import numpy as np
import pandas as pd

from utils import DataGenerator


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


class TestDataGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.df1, _ = dataset_generator()
        self.save = False
        self.datagenerator = DataGenerator(
            self.df1,
            "col_0",
            "datetime",
            "group",
            "nparray",
            "mp3s"
        )
        return super().setUp()
    
    def test_xy_shape(self):
        cols = self.datagenerator.cols
        for col in cols:
            y, x, _ = self.datagenerator.generate(col)
            print(y.shape, x.shape)
            self.assertEqual(y.shape, x.shape)

    def test_date_sequence(self):
        pass