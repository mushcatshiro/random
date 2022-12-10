import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor

from .base import BaseModel
from utils import CHKPT


class LinearModel(BaseModel):
    def __init__(
        self, X, y, metadata, threshold: float=None, topn: int=None, chkpt: CHKPT=None
    ):
        super().__init__()
        """
        TODO
        to validate if single model object instantiation is enough
        """
        self.X = X
        self.y = y
        self.metadata = metadata
        # self.mode = mode
        self.topn = topn
        self.threshold = threshold
        self.model = self.determine_run()
        self.chkpt = chkpt
        # TODO figure out a more elagant way of handling modelname
        self.modelname = None if not hasattr(metadata, "modelname") else metadata["modelname"]

    def determine_run(self):
        print(self.metadata["normality"])
        if self.metadata["normality"] or self.metadata["force_normal"]:
            return LinearRegression()
        elif not self.metadata["normality"] or self.metadata["force_robust"]:
            return HuberRegressor()
        elif self.metadata["load_model"]:
            self.modelname = self.metadata["load_model"]["modelname"]
            self.model = self.chkpt.retrieve_model(self.modelname)
        else:
            raise ValueError(
                "unable to determine run with the metadata provided"
            )

    def run(self, validate=False):
        reg = self.model.fit(self.X, self.y)
        self.report +=\
            "initial fitting:\n"\
            f"score: {reg.score(self.X, self.y)}\n"\
            f"coef: {reg.coef_}\n"\
            f"intecept: {reg.intercept_}\n"
        self.modelname =\
            f"{self.metadata['X']}_{self.metadata['y']}_{self.model.__class__.__name__}"
        print(self.report)
        print(self.modelname)
        # if self.topn:
        #     idx = np.argpartition(abs(reg.coef_), -self.topn).reshape(-1)[:-self.topn]
        #     drop_list = np.zeros(reg.coef_.shape)
        #     np.put(drop_list, idx, 1)
        #     drop_list = drop_list.astype(np.bool)
        #     print(f"topn drop list: {drop_list}")
        # elif self.threshold:
        #     drop_list = (abs(reg.coef_) < self.threshold).reshape(-1)
        #     print(f"th drop list: {drop_list}")
        # if validate:
        #     X = X.loc[:, drop_list]
        #     reg2 = self.model.fit(X, y)
        #     self.report +=\
        #         "2nd fitting:\n"\
        #         f"score: {reg2.score(X, y)}\n"\
        #         f"coef: {reg2.coef_}\n"
        #     if reg2.score(X, y) > reg.score(X, y):
        #         self.report +=\
        #             "2nd fitting score > initial fitting score, "\
        #             "recommend to investigate"
        return self

    def visualize(self, save=False):
        pred = self.model.predict(self.X)
        plt.scatter(self.X, self.y)
        plt.plot(
            self.X,
            pred,
            label=f"y={round(self.model.coef_[0], 2)}x+{round(self.model.intercept_, 2)}"
        )
        plt.plot(self.X, [0] * len(self.X), 'b')
        plt.plot([0] * len(self.y), self.y, 'b')
        plt.legend()
        plt.show()
        if save:
            pass
    
    def save_model(self):
        if self.modelname is not None:
            self.chkpt.save_model(self.modelname, self.model)
        else:
            raise ValueError("modelname is not found")

    def generate_report(self, dir):
        pass