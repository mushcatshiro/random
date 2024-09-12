import os
import unittest

from bootstrap import LinearModel
from utils import DataGenerator, CHKPT
from dataset_generator import dataset_generator


class TestLinearModel(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset, _ = dataset_generator()
        self.datagenerator = DataGenerator(
            self.dataset,
            "col_0",
            "datetime",
            "group",
            "nparray",
            "mp3s"
        )
        self.db = CHKPT(os.getcwd(), "testdb")
        self.cleanup_dir = None
        return super().setUp()
    
    def tearDown(self) -> None:
        # delete report
        if self.cleanup_dir is not None:
            import time
            time.sleep(2)
            os.rmdir(self.cleanup_dir)
        # delete db
        return super().tearDown()
    
    def test_default_model(self):
        for col in self.datagenerator.cols:
            X, y, _ = self.datagenerator.generate(col)
            m = self.datagenerator.metadata[col]
            m["X"] = col
            m["y"] = self.datagenerator.ycol
            model = LinearModel(
                X=X,
                y=y,
                metadata=m,
                chkpt=self.db,
                run_name="testlinear"
            )
            model.run()
            # model.generate_report()
            model.save_model()
            self.modelname = model.modelname
            # self.assertTrue()
        self.cleanup_dir = model.report_dir
    
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