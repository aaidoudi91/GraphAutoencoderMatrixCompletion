"""ALS on the residuals of a bias model, as in the team notebook (`als_bias`).

Stage 1, bias model:   B[u, i] = mean + user_bias[u] + item_bias[i]
Stage 2, ALS:          factorise the residuals R - B on the observed entries
Prediction:            R_hat[u, i] = B[u, i] + user_factors[u] . item_factors[i]

The table is read as table[user, item] with NaN for a missing rating, like the
rest of this project. The notebook stores movies in rows and 0 for missing;
the computations are the same, in the same order (items before users).
"""

import numpy as np


class ALSRecommender:
    def __init__(self, n_factors=20, lam=10.0, n_iters=15, lam_bias=10.0,
                 n_bias_iters=10, random_state=0, zero_unrated=False):
        if lam <= 0 or lam_bias <= 0:
            raise ValueError("lam and lam_bias must be strictly positive")
        self.n_factors = n_factors          # k
        self.lam = lam                      # penalty on both factor matrices
        self.n_iters = n_iters              # ALS passes
        self.lam_bias = lam_bias            # penalty of the bias model (lam_b in the notebook)
        self.n_bias_iters = n_bias_iters    # passes of the bias model
        self.random_state = random_state
        self.zero_unrated = zero_unrated    # False = as in the notebook (see fit)

    def fit(self, table):
        """table: (n_users, n_items), NaN where the rating is missing."""
        ratings = np.asarray(table, dtype=np.float64)
        m = np.isfinite(ratings)                               # observed entries
        R = np.where(m, ratings, 0.0)
        self.n_users, self.n_items = R.shape
        self.rating_min, self.rating_max = R[m].min(), R[m].max()
        k = self.n_factors

        # ---- Stage 1: global mean + item bias + user bias (shrunk averages)
        self.global_mean = R[m].mean()
        self.item_bias = np.zeros(self.n_items)
        self.user_bias = np.zeros(self.n_users)
        for _ in range(self.n_bias_iters):
            self.item_bias = (((R - self.global_mean - self.user_bias[:, None]) * m).sum(0)
                              / (self.lam_bias + m.sum(0)))
            self.user_bias = (((R - self.global_mean - self.item_bias[None, :]) * m).sum(1)
                              / (self.lam_bias + m.sum(1)))
        B = self.global_mean + self.user_bias[:, None] + self.item_bias[None, :]

        # ---- Stage 2: ALS on the residuals E = R - B (zero where not observed)
        E = (R - B) * m
        rng = np.random.RandomState(self.random_state)
        self.item_factors = 0.1 * rng.randn(self.n_items, k)   # I, drawn first as in the notebook
        self.user_factors = 0.1 * rng.randn(self.n_users, k)   # U
        identity = self.lam * np.eye(k)

        for _ in range(self.n_iters):
            for i in range(self.n_items):                      # items, with U fixed
                idx = np.where(m[:, i])[0]
                if len(idx):
                    U = self.user_factors[idx]
                    self.item_factors[i] = np.linalg.solve(U.T @ U + identity, U.T @ E[idx, i])
            for u in range(self.n_users):                      # users, with the new I fixed
                idx = np.where(m[u])[0]
                if len(idx):
                    I = self.item_factors[idx]
                    self.user_factors[u] = np.linalg.solve(I.T @ I + identity, I.T @ E[u, idx])

        # The notebook leaves users and items without ratings at their random
        # starting vectors. zero_unrated=True sets them to zero instead, so
        # their prediction is exactly the bias model.
        if self.zero_unrated:
            self.item_factors[~m.any(axis=0)] = 0.0
            self.user_factors[~m.any(axis=1)] = 0.0
        return self

    def _clip(self, pred, clip):
        return np.clip(pred, self.rating_min, self.rating_max) if clip else pred

    def predict(self, users, items, clip=True):
        """Predicted ratings for paired arrays of user and item indices."""
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        pred = (self.global_mean + self.user_bias[users] + self.item_bias[items]
                + np.sum(self.user_factors[users] * self.item_factors[items], axis=-1))
        return self._clip(pred, clip)

    def predict_all(self, clip=True):
        """Predicted rating of every pair: (n_users, n_items)."""
        pred = (self.global_mean + self.user_bias[:, None] + self.item_bias[None, :]
                + self.user_factors @ self.item_factors.T)
        return self._clip(pred, clip)

    def predict_bias_only(self, users, items, clip=True):
        """The stage-1 prediction alone: mean + user bias + item bias."""
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        return self._clip(self.global_mean + self.user_bias[users] + self.item_bias[items], clip)