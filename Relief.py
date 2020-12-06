import os
import sys
from functools import reduce
from collections import deque
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Relief(BaseEstimator, TransformerMixin):

    def __init__(self):
        kwargs = dict(kwargs)
        self.w_ = None

        def gen_random_state(rnd_state):
            """Generate random state instance"""
            if isinstance(rnd_state, np.random.RandomState):
                return rnd_state

            return np.random.RandomState(seed=rnd_state)

        for name, default_value, convf in (
                # Param name, default param value, param conversion function
                ('categorical', (), tuple),
                ('n_jobs', os.cpu_count(), int),
                ('n_iterations', 100, int),
                ('n_features', 1, int),
                ('random_state', None, gen_random_state)
        ):
            setattr(self, name, convf(kwargs.setdefault(name, default_value)))
            del kwargs[name]

    def fit(self, data, y):
        n, m = data.shape # Number of instances & features

        # Initialise state
        js = self.random_state.randint(n, size=self.n_iterations)

        self.w_ = self._fit_iteration(data, y, 0, self.n_iterations, js)

        self.w_ /= self.n_iterations

        return self

    def _fit_iteration(self, data, y, iter_offset, n_iters, js):
        w = np.array([0.] * data.shape[1])

        for i in range(iter_offset, n_iters + iter_offset):
            j = js[i]
            ri = data[j] # Random sample instance
            hit, miss = self._nn(data, y, j)

            w += np.array([
                self._diff(k, ri[k], miss[k]) - self._diff(k, ri[k], hit[k])
                for k in range(data.shape[1])
            ])

        return w

    def _nn(self, data, y, j):
        ri = data[j]
        d = np.sum(
            np.array([
                self._diff(c, ri[c], data[:, c]) for c in range(len(ri))
            ]).T,
            axis=1
        )

        odata = data[d.argsort()]
        oy = y[d.argsort()]

        h = odata[oy == y[j]][0:1]
        m = odata[oy != y[j]][0]

        h = h[1] if h.shape[0] > 1 else h[0]

        return h, m

    def _diff(self, c, a1, a2):
        return (
            np.abs(a1 - a2) if c not in self.categorical
            else 1 - (a1 == a2)
        )

    def transform(self, data):
        feat_indices = np.flip(np.argsort(self.w_), 0)[0:self.n_features]
        return data[:, feat_indices]
