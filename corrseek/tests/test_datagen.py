import unittest

import numpy as np
import pandas as pd

from utils import DataGenerator
from tests.dataset_generator import dataset_generator, fast_skewed_dataset_generator


class TestDataGenerator(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def test_xy_shape(self):
        df1, _ = dataset_generator()
        save = False
        datagenerator = DataGenerator(
            df1,
            "col_0",
            "datetime",
            "group",
            "nparray",
            "mp3s"
        )
        for col in datagenerator.cols:
            y, x, _ = datagenerator.generate(col)
            print(y.shape, x.shape)
            self.assertEqual(y.shape, x.shape)
    
    def test_non_normal_dist(self):
        df = fast_skewed_dataset_generator(100, -3)
        datagenerator = DataGenerator(
            df,
            "col_0",
            "datetime",
            "group",
            "nparray",
            "mp3s"
        )
        self.assertFalse(datagenerator.metadata["col_1"]["normality"])
        self.assertTrue(datagenerator.metadata["col_0"]["normality"])

    def test_date_sequence(self):
        pass