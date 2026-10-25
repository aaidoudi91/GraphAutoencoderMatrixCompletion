import numpy as np
import torch


class RatingGraphs:
    """One normalised adjacency matrix per rating level.

    Nodes are numbered users first, then items:
        user u -> node u
        item i -> node n_users + i
    For each level t, adj[t] is the sparse (N x N) matrix D^-1/2 A_t D^-1/2,
    where A_t has a 1 for every (user, item) pair rated at level t.
    Build it from the TRAINING ratings only.
    """

    def __init__(self, users, items, level_idx, n_users, n_items, T, device=None):
        self.n_users = n_users
        self.n_items = n_items
        self.N = n_users + n_items
        self.T = T
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        level_idx = np.asarray(level_idx, dtype=np.int64)

        self.adj = []      # T sparse tensors (N x N)
        self.degree = []   # T arrays of length N: number of neighbours in G_t
        for t in range(T):
            sel = level_idx == t
            u = users[sel]
            i = items[sel] + n_users         

            rows = np.concatenate([u, i])
            cols = np.concatenate([i, u])

            deg = np.bincount(rows, minlength=self.N)
            values = 1.0 / np.sqrt(deg[rows] * deg[cols])

            adj = torch.sparse_coo_tensor(
                torch.as_tensor(np.stack([rows, cols])),
                torch.as_tensor(values, dtype=torch.float32),
                size=(self.N, self.N),
            ).coalesce().to(self.device)

            self.adj.append(adj)
            self.degree.append(deg)

    def n_edges(self):
        """Number of ratings in each graph."""
        return [int(d[: self.n_users].sum()) for d in self.degree]

    def propagate(self, t, x):
        """One round of message passing in graph t: returns adj[t] @ x."""
        return torch.sparse.mm(self.adj[t], x)