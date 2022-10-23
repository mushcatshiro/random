from scipy.stats import f_oneway, kruskal


class AnovaTest:
    """
    scipy.stats.f_oneway for all distribution that are normal
    scipy.stats.kruskal for any distribution that is non normal
    """
    def __init__(self) -> None:
        pass

    def test_normality(self):
        pass

    def run(self, X, group):
        g1, remaining = self.test_normality(X, group)
        if g1 and all(remaining):
            f_oneway()
        else:
            kruskal()
        pass