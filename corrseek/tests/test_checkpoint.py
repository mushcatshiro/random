import os
import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from utils import CHKPT


X = np.random.rand(100)
y1 = X + np.random.rand(100)
model1 = LinearRegression().fit(
    X=X.reshape(-1, 1),
    y=y1.reshape(-1, 1)
)
model2 = LinearRegression().fit(
    X=X.reshape(-1, 1),
    y=X
)


class TestCHKPT(unittest.TestCase):
    def setUp(self) -> None:
        self.db: CHKPT = CHKPT(
            os.getenv('CORRSEEKDBPATH') or os.getcwd(),
            os.getenv('CORESEEKDBNAME') or 'corrseekchkpt'
        )
        self._db_abs_dir = self.db.db_abs_dir
        return super().setUp()
    
    def tearDown(self) -> None:
        os.remove(self._db_abs_dir)
        return super().tearDown()

    def _test_save_new_model(self):
        self.db.save_model("modelname", model1)
        ret:LinearRegression = self.db.retrieve_model("modelname")
        self.assertEqual(ret.get_params(), model1.get_params())

    def _test_update_model(self):
        self.db.save_model("modelname", model2)
        ret:LinearRegression = self.db.retrieve_model("modelname")
        self.assertEqual(ret.get_params(), model2.get_params())
    
    def test_run(self):
        self._test_save_new_model()
        self._test_update_model()