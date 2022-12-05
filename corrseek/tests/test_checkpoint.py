import os
import unittest

from utils import CHKPT


payload = {
    "model1": {
        "modelname": "testmodel1",
        "parameters": {"alpha": 1, "beta": 2}
    },
    "model2": {
        "modelname": "testmodel2",
        "parameters": {"alpha": 0.5, "beta": 13}
    }
}


class TestCHKPT(unittest.TestCase):
    def setUp(self) -> None:
        self.db = CHKPT(
            os.getenv('CORRSEEKDBPATH') or os.getcwd(),
            os.getenv('CORESEEKDBNAME') or 'corrseekchkpt'
        )
        self._db_abs_dir = self.db.db_abs_dir
        return super().setUp()
    
    def tearDown(self) -> None:
        os.remove(self._db_abs_dir)
        return super().tearDown()

    def _test_save_new_model(self):
        self.db.save_model(payload["model1"])
        ret = self.db.retrieve_model(payload["model1"]["modelname"])
        self.assertEqual(ret, payload["model1"]["parameters"])

    def _test_update_model(self):
        ret = self.db.save_model(payload["model2"])
        self.assertEqual(ret, payload["model1"]["parameters"])
    
    def test_run(self):
        self._test_save_new_model()
        self._test_update_model