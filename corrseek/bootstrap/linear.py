import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .base import BaseModel
from utils import CHKPT


class linearModel(BaseModel):
    def __init__(
        self, X, y, metadata, chkpt: CHKPT, run_name, report_dir=None
    ):
        self.X = X
        self.y = y
        self.metadata:dict = metadata
        super().__init__(chkpt=chkpt, run_name=run_name, report_dir=report_dir)

    def run(self):
        reg = self.model.fit(self.X, self.y)
        score = reg.score(self.X, self.y)
        if score < 0.2:
            fit = "good"
        elif score > 0.2 and score < 0.7:
            fit = "moderate"
        elif score > 0.7:
            fit = "bad"
        self.report +=\
            "model learnt:\n"\
            f"score: {score}\n"\
            f"fit: {fit}\n"\
            f"coef: {reg.coef_}\n"\
            f"intecept: {reg.intercept_}\n"
        self.modelname =\
            f"{self.metadata['X']}_{self.metadata['y']}_{self.model.__class__.__name__}"
        print(self.report)
        print(self.modelname)
        return self


class LinearModel(linearModel):
    def determine_run(self):
        if self.X.shape[1] > 1 or self.y.shape[1] > 1:
            raise ValueError("expects 1D for X and y")
        if self.metadata["normality"] or self.metadata["force_normal"]:
            return LinearRegression()
        elif not self.metadata["normality"] or self.metadata["force_robust"]:
            return HuberRegressor()
        elif self.metadata["load_model"]:
            self.modelname = self.metadata["load_model"]["modelname"]
            model = self.chkpt.retrieve_model(self.modelname)
            if not model:
                raise ValueError(
                    f"model {self.modelname} "
                    "does not exists in checkpoint database"
                )
            self.model = model
        else:
            return LinearRegression()


class MLinearModel(linearModel):
    def determine_run(self):
        if not isinstance(self.X, pd.DataFrame):
            pass
        if len(self.X.shape) == 1 or len(self.y.shape) > 1:
            raise ValueError("expects 2D for X and 1D for y")
        if self.metadata["force_normal"]:
            return LinearRegression()
        elif self.metadata["force_robust"]:
            return HuberRegressor()
        elif self.metadata["load_model"]:
            self.modelname = self.metadata["load_model"]["modelname"]
            self.model = self.chkpt.retrieve_model(self.modelname)
        else:
            return LinearRegression()
    
    def run(self):
        vif = pd.DataFrame()
        vif["feature"] = self.X.columns
        vif["VIF"] = [
            variance_inflation_factor(self.X.values, i)
            for i in range(len(self.X.columns))
        ]
        self.report += vif
        self.report += "\n"
        return super().run()