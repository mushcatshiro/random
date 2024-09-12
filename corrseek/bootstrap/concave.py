import numpy as np

from .base import BaseModel


class ConcavityModel(BaseModel):
    """
    to detect steep concave upward/downwards in dataset

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
        split = len(X)//2
        if len(X.shape) > 1:
        
            g = (y[split:] - y[:-split])/(X[:, split:] - X[:, :-split])
        else:
            g = (y[split:] - y[:-split])/(X[split:] - X[:-split])
        i = np.isinf(g)
        g[i] = 0
        return g

    def _moving_average(self, arr, n):
        ret = np.cumsum(arr, axis=1)
        ret[:, n:] = ret[:, n:] - ret[:, :-n]
        return ret[:, n-1:]/n

    def _cluster_approach(self, X, y, n):
        c = np.asarray([X, y])
        cluster = np.average(c.reshape(-1, n, 2), axis=1)
        return cluster

    def _get_threshold_idx(self, arr):
        idx = np.where(arr > self.threshold, True, False)
        if len(idx.shape) > 1:
            idx = np.pad(idx, [(0, 0), (1, 0)], mode='constant')
        else:
            idx = np.pad(idx, (1, 0), mode='constant')
        return idx

    def visualize(self):
        pass

    def run(self, X, y):
        # TODO
        # axis 0 or 1 to run in parallel
        g = self._pairwise_gradient(X, y)
        print(g)
        if self.threshold:
            idx = self._get_threshold_idx(g)
            print(idx)
        else:
            # TODO find 99th percentile? or max?
            raise
        # TODO check idx
        self.report += f"cliff found at position {idx} with gradient {g}"
        return self

# m = ConcavityModel(sigma=1, threshold=2)
# y = np.asarray([1, 2, 3, 4, 8, 9, 9])
# X = np.asarray([[1, 2, 3, 4, 5, 6, 7], [1, 1, 2, 3, 4, 5, 6]])
# X = np.asarray([1, 2, 3, 4, 5, 6, 7])
# print(X.shape)
# m.run(X=X, y=y)
# raw = np.cumsum(np.random.normal(5, 100, 1000))
# print(raw.shape)
# smooth = gaussian_filter1d(raw, 100)
# smooth_d2 = np.gradient(np.gradient(smooth))
