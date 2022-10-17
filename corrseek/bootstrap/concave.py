import numpy as np
from scipy.ndimage import gaussian_filter1d

# from .base import BaseModel


class ConcavityModel:
    """
    threshold mode: might return multiple values
    non-threshold mode: return max cliff
    """
    def __init__(self, sigma, threshold):
        super().__init__()
        self.sigma = sigma
        self.threshold = threshold
        self.report = ""
    
    def _gaussian_filter_id(self, size):
        filter_range = np.linspace(-int(size/2),int(size/2),size)
        gaussian_filter = [
            1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2/(2 * self.sigma**2))  # noqa
            for x in filter_range
        ]
        return gaussian_filter
    
    def _pairwise_gradient(self, X, y):
        if len(X.shape) > 1:
            g = (y[1:] - y[:-1])/(X[:, 1:] - X[:, :-1])
        else:
            g = (y[1:] - y[:-1])/(X[1:] - X[:-1])
        i = np.isinf(g)
        g[i] = 0
        return g
    
    def _threshold_idx(self, arr):
        idx = np.where(arr > self.threshold, True, False)
        if len(idx.shape) > 1:
            idx = np.pad(idx, [(0, 0), (1, 0)], mode='constant')
        else:
            idx = np.pad(idx, (1, 0), mode='constant')
        return idx

    def run(self, X, y):
        g = self._pairwise_gradient(X, y)
        print(g)
        if self.threshold:
            idx = self._threshold_idx(g)
            print(idx)
        else:
            # TODO find 99th percentile? or max?
            pass
        # TODO check idx
        self.report += f"cliff found at position {idx} with gradient {g}"
        return self

m = ConcavityModel(sigma=1, threshold=2)
y = np.asarray([1, 2, 3, 4, 8, 9, 9])
# X = np.asarray([[1, 2, 3, 4, 5, 6, 7], [1, 1, 2, 3, 4, 5, 6]])
X = np.asarray([1, 2, 3, 4, 5, 6, 7])
print(X.shape)
m.run(X=X, y=y)
# raw = np.cumsum(np.random.normal(5, 100, 1000))
# print(raw.shape)
# smooth = gaussian_filter1d(raw, 100)
# smooth_d2 = np.gradient(np.gradient(smooth))
