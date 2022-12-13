from abc import ABC, abstractmethod
import datetime as dt
import os

from utils import CHKPT

class BaseModel(ABC):
    def __init__(self, chkpt: CHKPT, run_name, report_dir=None):
        self.report = ""
        self.modelname = None
        self.run_name = run_name
        self.report_dir = self._report_dir(report_dir)
        self.chkpt:CHKPT = chkpt
        self.model = self.determine_run()
    
    def _report_dir(self, report_dir):
        if report_dir is None:
            report_dir = os.path.join(
                os.getcwd(),
                f"{self.run_name}"
            )
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)
        return report_dir
    
    @abstractmethod
    def determine_run(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def save_model(self):
        if self.modelname is not None and self.chkpt:
            self.chkpt.save_model(self.modelname, self.model)
        else:
            raise ValueError("modelname is not found")

    def generate_report(self):
        if self.report:
            with open(
                os.path.join(self.report_dir, self.modelname)
                , "w"
            ) as wf:
                wf.write(self.report)
        else:
            raise ValueError(
                "report is empty possibly due to incorrect sequence of execution"
            )