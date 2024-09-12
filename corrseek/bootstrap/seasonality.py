from statsmodels.tsa.stattools import adfuller, kpss

# from .base import BaseModel


def test_seasonality(X, y, robust: bool=False):
    """
    test seasonality is tested at 5%
    adf test null hypothesis assumes there is a unit root,
    or in other words is non-stationary + not necessaryly have trend
    """
    disagree = None
    result = adfuller(X, autolag="AIC")
    adfresult = {
        "adfstat": result[0],
        "pvalue": result[1],
        "usedlag": result[2],
        "nobs": result[3],
        "critcalues": result[4],
        "icbest": result[5]
    }
    ret = False if adfresult["pvalue"] >= 0.05 else True
    if robust:
        """
        kpss null hypothesis assumes time series is stationary
        """
        t = kpss(X)
        kpssresult = {
            "kpssstat": t[0],
            "pvalue": t[1],
            "nlags": t[2],
            "crit_dict": t[3]
        }
        robust_ret = True if kpssresult["pvalue"] <= 0.05 else False
        if robust_ret != ret:
            disagree = True
    return ret, disagree