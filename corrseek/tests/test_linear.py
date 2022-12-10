import os
import unittest

from bootstrap import LinearModel
from utils import DataGenerator, CHKPT
from dataset_generator import dataset_generator


REPORTDIR = os.path.join(os.getcwd(), 'testlinear')


class TestLinearModel(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset, _ = dataset_generator()
        self.datagenerator = DataGenerator(
            self.dataset,
            "col_0",
            "datatime",
            "group",
            "nparray",
            "mp3s"
        )
        os.mkdir(REPORTDIR)
        if not os.path.exists(REPORTDIR):
            raise FileNotFoundError
        self.db = CHKPT(os.getcwd(), "testdb")
        return super().setUp()
    
    def tearDown(self) -> None:
        # delete report
        os.remove(REPORTDIR)
        # delete db
        return super().tearDown()
    
    def _test_default_model(self):
        for col in self.datagenerator.cols:
            X, y, _ = self.datagenerator.generate(col)
            m = self.datagenerator.metadata[col]
            m["X"] = col
            m["y"] = self.datagenerator.ycol
            model = LinearModel(
                X=X,
                y=y,
                metadata=m,
                threshold=1,
                chkpt=self.db
            )
            model.run()
            model.generate_report(REPORTDIR)
            model.save_model()
            self.modelname = model.modelname
            # self.assertTrue()
    
    def test_force_model(self):
        pass

    def test_save_model(self):
        pass

    def _test_load_model(self):
        for col in self.datagenerator.cols:
            X, y, _ = self.datagenerator.generate(col)
            m = self.datagenerator.metadata[col]
            m["X"] = col
            m["y"] = self.datagenerator.ycol
            m["load_model"] = {"modelname": f"{col}_{self.datagenerator.ycol}_{self.modelname}"}