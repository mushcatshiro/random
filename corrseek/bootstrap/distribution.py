from scipy.stats import spearmanr, wilcoxon, mannwhitneyu

from .base import BaseModel
from utils import CHKPT


class DistMetaModel(BaseModel):
    def __init__(
        self, X, chkpt: CHKPT, run_name, y = None, metadata=None, report_dir=None
    ):
        self.X = X
        self.y = y
        self.metadata = None
        super().__init__(chkpt, run_name, report_dir)

    def determine_run(self):
        if self.X.shape[1] > 1 or self.y.shape[1] > 1:
            raise ValueError(
                "the model only attempts to study pairwise correlation "
                "between independent and dependent variable"
            )
    
    def run(self):
        # determine if two variables are monotonic
        corr, pvalue = spearmanr(self.X, self.y)
        self.metadata["monotonic"] = True if pvalue < 0.05 else False
        corr, pvalue = None, None
        # determine if two variables are symmetric
        # to validate with real data the intuition above seems wrong
        # known as non-parametric paired t-test
        _, pvalue, _ = wilcoxon(self.X. self.y)
        self.metadata["symmetric"] = True if pvalue < 0.05 else False
        pvalue = None
        # determine if two variables are
        # from identical distribution/stocastically equivalent
        _, pvalue = mannwhitneyu(self.X, self.y)
        self.metadata["iid"] = True if pvalue > 0.05 else False
        pvalue = None
        return self