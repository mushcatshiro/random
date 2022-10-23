import warnings

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from .base import BaseModel


class PolynomialModel(BaseModel):
    def __init__(self, order=2, interaction=True):
        super().__init__()
        self.model = LinearRegression()
        if order > 2:
            warnings.warn(
                "order higher than 2 is not recommended "
                "as overfitting is likely happen"
            )
        self.preprocessing = PolynomialFeatures(
            degree=order, interaction_only=interaction
        )

    def run(self, X, y):
        hoX = self.preprocessing.fit_transform(X)
        reg = self.model.fit(hoX, y)
        return self