from statsmodels.tsa.stattools import adfuller, kpss

# from .base import BaseModel


def test_seasonality(X, y, robust: bool=False):
    result = adfuller(X, autolag="AIC")
    if robust:
        t = kpss(X)
        result = ""
    print(result)
    return