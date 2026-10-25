import numpy as np

class RatingData:
    """Holds the rating matrix and everything derived from it."""

    def __init__(self, table):
        
        self.table = table
        self.n_users, self.n_items = table.shape
        self.mask = ~np.isnan(table)

        self.users, self.items = np.nonzero(self.mask)
        self.ratings = table[self.mask]
        
        #self.users[i] users of the raiting i
        #self.items[i] items of the raiting i  
        #self.ratings[i] ratings of the raiting i  
        
        self.levels = np.unique(self.ratings)
        self.T = len(self.levels)
        self.level_idx = np.searchsorted(self.levels, self.ratings)

        self.global_mean = self.ratings.mean()

    def split(self, val_ratio=0.1, seed=0):
        """Random split of the observed ratings into train and validation.

        Every user and item that has ratings keeps at least one in the
        training part. Returns two tuples (users, items, level_idx, ratings).
        """
        rng = np.random.default_rng(seed)
        n = len(self.ratings)
        perm = rng.permutation(n)
        n_val = int(val_ratio * n)
        is_val = np.zeros(n, dtype=bool)
        is_val[perm[:n_val]] = True

        train_per_user = np.bincount(self.users[~is_val], minlength=self.n_users)
        train_per_item = np.bincount(self.items[~is_val], minlength=self.n_items)
        for k in np.nonzero(is_val)[0]:
            u, i = self.users[k], self.items[k]
            if train_per_user[u] == 0 or train_per_item[i] == 0:
                is_val[k] = False
                train_per_user[u] += 1
                train_per_item[i] += 1

        def triples(sel):
            return (self.users[sel], self.items[sel],
                    self.level_idx[sel], self.ratings[sel])

        return triples(~is_val), triples(is_val) #here is the triplet who interesting us 