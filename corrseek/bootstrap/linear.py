import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LinearRegression

from .base import BaseModel


class LinearModel(BaseModel):
    def __init__(self, threshold: float=None, topn: int=None):
        super().__init__()
        """
        TODO
        to validate if single model object instantiation is enough
        """
        self.model = LinearRegression()
        self.topn = topn
        self.threshold = threshold
        assert self.topn or self.threshold, "topn or threshold must be set"

    def run(self, X:DataFrame, y:DataFrame, validate=False):
        reg = self.model.fit(X, y)
        self.report +=\
            "initial fitting:\n"\
            f"score: {reg.score(X, y)}\n"\
            f"coef: {reg.coef_}\n"\
            f"intecept: {reg.intercept_}\n"
        if self.topn:
            idx = np.argpartition(abs(reg.coef_), -self.topn).reshape(-1)[:-self.topn]
            drop_list = np.zeros(reg.coef_.shape)
            np.put(drop_list, idx, 1)
            drop_list = drop_list.astype(np.bool)
            print(f"topn drop list: {drop_list}")
        elif self.threshold:
            drop_list = (abs(reg.coef_) < self.threshold).reshape(-1)
            print(f"th drop list: {drop_list}")
        if validate:
            X = X.loc[:, drop_list]
            reg2 = self.model.fit(X, y)
            self.report +=\
                "2nd fitting:\n"\
                f"score: {reg2.score(X, y)}\n"\
                f"coef: {reg2.coef_}\n"
            if reg2.score(X, y) > reg.score(X, y):
                self.report +=\
                    "2nd fitting score > initial fitting score, "\
                    "recommend to investigate"
        return self

    def visualize(self, X, y, save=False):
        # TODO changing to sns for ease of mind
        for col in X.columns:
            reg = self.model.fit(X[col].to_numpy().reshape(-1, 1), y)
            pred = reg.predict(X[col].to_numpy().reshape(-1, 1))
            plt.scatter(X[col], y)
            plt.plot(X[col], pred, label=f"y={round(reg.coef_[0], 2)}x+{round(reg.intercept_, 2)}")
            plt.plot(X[col], [0] * len(X[col]), 'b')
            plt.plot([0] * len(y), y, 'b')
            plt.legend()
            plt.show()
            if save:
                pass