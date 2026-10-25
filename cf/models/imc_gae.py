import torch
import torch.nn as nn


class NodeEmbedding(nn.Module):
    """Starting vectors of the nodes:

        x_t^0[v] = [ a | rho_role(v) | feature pieces | l_t[v] ]

    a          one vector shared by every node            (identical feature)
    rho        one vector for users, one for items        (role-aware feature)
    features   one piece per feature set, e.g. combined genre/year or title (optional)
    l_t[v]     one vector per node and per rating level   (personal embedding)

    Each feature set is a fixed matrix F of shape (N, n_k), with one row per
    node (zeros for the nodes that do not have the feature, i.e. all users),
    and gets trainable weights W. By default W has shape (T, n_k, d_k), so
    each rating level has its own projection. With share_feature_weights=True,
    W has shape (n_k, d_k) and the same projected feature is used at every
    rating level. Its piece is

        piece_t[v] = F[v] @ W[t]   (separate) or F[v] @ W (shared)

    A feature matrix can contain binary/categorical columns or numeric values.
    For example, the notebook combines normalized genre weights and one
    standardized release-year value in a single matrix; title embeddings can
    remain a separate matrix. Rows of zeros give zeros.

    features   None, a dict {name: matrix} (recommended, the names are kept),
               or a list of matrices (named "feat0", "feat1", ...).
    d_feat     size of each piece: one int for all, or a dict {name: int}.
    share_feature_weights  whether each feature set uses one projection for
                           all rating levels instead of one projection per level.

    Nodes are numbered users first, then items, as in RatingGraphs.
    """

    def __init__(self, n_users, n_items, T, d_shared=16, d_role=16, d_personal=16,
                 features=None, d_feat=8, share_feature_weights=False):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.N = n_users + n_items
        self.T = T
        self.d_shared, self.d_role, self.d_personal = d_shared, d_role, d_personal
        self.share_feature_weights = share_feature_weights

        self.shared = nn.Parameter(torch.empty(d_shared))                  # (d_shared,)
        self.role = nn.Embedding(2, d_role)                                # (2, d_role)   0 = user, 1 = item
        self.personal = nn.Parameter(torch.empty(T, self.N, d_personal))   # (T, N, d_personal)

        # Role of each node: 0 for the first n_users nodes, 1 for the rest.
        role_of_node = torch.cat([
            torch.zeros(n_users, dtype=torch.long),
            torch.ones(n_items, dtype=torch.long),
        ])
        self.register_buffer("role_of_node", role_of_node)                 # (N,)

        # Feature sets, in a fixed order
        if features is None:
            named = {}
        elif isinstance(features, dict):
            named = dict(features)
        elif isinstance(features, (list, tuple)):
            named = {f"feat{k}": f for k, f in enumerate(features)}
        else:
            named = {"feat0": features}
        self.feature_names = tuple(named)                                  # e.g. ("genre_year", "title")
        self.n_feature_sets = len(named)
        self.feature_dims = []                                             # size of each piece
        self.feat = nn.ParameterList()
        for k, (name, f) in enumerate(named.items()):
            size = d_feat.get(name, 8) if isinstance(d_feat, dict) else d_feat
            f = torch.as_tensor(f, dtype=torch.float32)                    # (N, n_k)
            assert f.shape[0] == self.N, f"feature '{name}' needs one row per node"
            self.register_buffer(f"features_{k}", f)                       # fixed data, not trained
            shape = (f.shape[1], size) if share_feature_weights else (T, f.shape[1], size)
            w = nn.Parameter(torch.empty(*shape))
            nn.init.normal_(w, std=0.1)
            self.feat.append(w)
            self.feature_dims.append(size)
        self.d_feat = sum(self.feature_dims)
        self.dim = d_shared + d_role + self.d_feat + d_personal

        nn.init.normal_(self.shared, std=0.1)
        nn.init.normal_(self.role.weight, std=0.1)
        nn.init.normal_(self.personal, std=0.1)

    def pieces(self):
        """Where each piece sits in the starting vector: {name: slice of the last dimension}."""
        sizes = [("shared", self.d_shared), ("role", self.d_role)]
        sizes += list(zip(self.feature_names, self.feature_dims))
        sizes.append(("personal", self.d_personal))
        out, start = {}, 0
        for name, size in sizes:
            out[name] = slice(start, start + size)
            start += size
        return out

    def forward(self):
        """Returns x0 of shape (T, N, dim)."""
        shared = self.shared.expand(self.T, self.N, -1)                    # (T, N, d_shared)
        role = self.role(self.role_of_node).expand(self.T, self.N, -1)     # (T, N, d_role)
        parts = [shared, role]
        for k in range(self.n_feature_sets):
            f = getattr(self, f"features_{k}")                             # (N, n_k)
            if self.share_feature_weights:
                # One feature projection, broadcast into every rating-level graph.
                piece = f @ self.feat[k]                                   # (N, d_k)
                parts.append(piece.unsqueeze(0).expand(self.T, -1, -1))    # (T, N, d_k)
            else:
                # piece[t, n] = f[n] @ W[t], with a separate W for each level.
                parts.append(torch.einsum("ng,tgd->tnd", f, self.feat[k])) # (T, N, d_k)
        parts.append(self.personal)                                        # (T, N, d_personal)
        return torch.cat(parts, dim=-1)                                    # (T, N, dim)


class Encoder(nn.Module):
    """From the starting vectors x0 to the final node vectors n[v].

    For each layer l = 0 .. L-1:
        1. node dropout (training only): a random set of nodes sends zeros
        2. message passing in every rating graph: x_t^{l+1} = A_t @ x_t^l
        3. ReLU, except after the last layer
    Then:
        h_t[v] = sum_l x_t^l[v] / (l + 1)      weighted sum over layers
        h[v]   = sum_t h_t[v]                  sum over rating levels
        n[v]   = tanh(W h[v])                  dense layer

    With layer_weights=True, step 2 and 3 become
        x_t^{l+1} = ReLU( (A_t @ x_t^l) @ W^l )
    with one learned (dim, dim) matrix W^l per layer, shared by all rating
    levels, and ReLU after every layer. W^l starts as the identity.

    With layer_weights=False, there is no W^l. Each raw propagated vector is
    passed through intermediate layers without ReLU. ReLU is applied only to
    the final graph-propagation output, which is then added to the aggregation.
    """

    def __init__(self, dim, out_dim=64, n_layers=2, p0=0.1, theta=0.05, layer_weights=False):
        super().__init__()
        self.n_layers = n_layers
        self.p0 = p0          # dropout probability at layer 0
        self.theta = theta    # change of that probability per layer
        self.dense = nn.Linear(dim, out_dim, bias=True)           # W: (out_dim, dim), plus output bias
        self.use_layer_weights = layer_weights
        if layer_weights:
            self.layer_w = nn.ParameterList(
                [nn.Parameter(torch.eye(dim)) for _ in range(n_layers)])   # each (dim, dim)

    def dropout_prob(self, layer):
        """p_l = p0 + l * theta, kept inside [0, 0.9]."""
        return min(max(self.p0 + layer * self.theta, 0.0), 0.9)

    def drop_nodes(self, x, p):
        """Zero the whole vector of a random set of nodes, rescale the others.

        x: (T, N, dim). The same nodes are dropped in every rating graph.
        """
        if not self.training or p <= 0:
            return x
        N = x.shape[1]
        keep = (torch.rand(N, 1, device=x.device) >= p).to(x.dtype)   # (N, 1)  1 = kept
        return x * keep / (1.0 - p)                                   # (T, N, dim)

    def forward(self, x0, graphs):
        """x0: (T, N, dim). Returns n: (N, out_dim) and h_t: (T, N, dim)."""
        T = x0.shape[0]
        x = x0                                                        # (T, N, dim)  x^0
        h_t = x0                                                      # (T, N, dim)  layer 0 has weight 1

        for layer in range(self.n_layers):
            sent = self.drop_nodes(x, self.dropout_prob(layer))       # (T, N, dim)  what nodes send
            propagated = torch.stack(
                [graphs.propagate(t, sent[t]) for t in range(T)]      # each (N, dim)
            )                                                         # (T, N, dim)  x^{l+1}
            if self.use_layer_weights:
                x = torch.relu(propagated @ self.layer_w[layer])      # transform + ReLU at every layer
                layer_output = x
            else:
                if layer == self.n_layers - 1:
                    x = torch.relu(propagated)                        # ReLU only after the final propagation
                else:
                    x = propagated                                    # pass raw output to the next layer
                layer_output = x
            h_t = h_t + layer_output / (layer + 2)                    # (T, N, dim)  weight 1/(l+1)

        h = h_t.sum(dim=0)                                            # (N, dim)
        n = torch.tanh(self.dense(h))                                 # (N, out_dim)
        return n, h_t


class Decoder(nn.Module):
    """From two node vectors to a probability per rating level.

        e_t(u, i) = n[u]^T  W_t  n[i]            one score per level
        p_t(u, i) = softmax over t of e_t        probabilities
        r_hat     = sum_t levels[t] * p_t        predicted rating
    """

    def __init__(self, dim, T):
        super().__init__()
        self.T = T
        self.weight = nn.Parameter(torch.empty(T, dim, dim))       # (T, k, k)  weight[t] = W_t
        with torch.no_grad():
            for t in range(T):
                nn.init.xavier_uniform_(self.weight[t])                # each W_t: (k, k)

    def forward(self, n_u, n_i):
        """n_u, n_i: (B, k), the vectors of B users and of B items, pair by pair.
        Returns the scores (logits) of shape (B, T)."""
        # for each pair b and level t:  sum_{k,l} n_u[b,k] * W[t,k,l] * n_i[b,l]
        return torch.einsum("bk,tkl,bl->bt", n_u, self.weight, n_i)   # (B, T)

    @staticmethod
    def expected_rating(logits, levels):
        """logits: (..., T), levels: (T,). Returns the predicted ratings (...)."""
        probs = torch.softmax(logits, dim=-1)                      # (..., T)  sums to 1 over t
        return (probs * levels).sum(dim=-1)                        # (...)

    @torch.no_grad()
    def predict_full(self, n_users, n_items, levels, chunk=512, extra=None):
        """Predicted rating of every (user, item) pair.

        n_users: (n_users, k), n_items: (n_items, k), levels: (T,).
        extra: optional (n_users, n_items, T) scores added before the softmax.
        Returns (n_users, n_items). Users are processed `chunk` at a time
        so the (chunk, n_items, T) scores fit in memory.
        """
        out = []
        for start in range(0, n_users.shape[0], chunk):
            block = n_users[start:start + chunk]                   # (c, k)
            logits = torch.einsum("uk,tkl,il->uit", block, self.weight, n_items)   # (c, n_items, T)
            if extra is not None:
                logits = logits + extra[start:start + chunk]       # (c, n_items, T)
            out.append(self.expected_rating(logits, levels))       # (c, n_items)
        return torch.cat(out, dim=0)                               # (n_users, n_items)


class GenreTerms(nn.Module):
    """Extra scores added to the decoder's:

        extra[u, i, t] = alpha_t * log s[u, i, t]  +  sum_g Gn[i, g] * b[g, t]

    s      fixed data: how user u rates the genres of film i   (n_users, n_items, T)
    Gn     fixed data: normalised genres of each film          (n_items, n_genres)
    alpha  learned: trust in the profile, one per level        (T,)
    b      learned: tendency of each genre at each level       (n_genres, T)

    alpha and b start at zero, so the model starts as the plain IMC-GAE.
    """

    def __init__(self, s, item_genres, eps=1e-6):
        super().__init__()
        s = torch.as_tensor(s, dtype=torch.float32)                        # (n_users, n_items, T)
        item_genres = torch.as_tensor(item_genres, dtype=torch.float32)    # (n_items, n_genres)
        assert s.shape[1] == item_genres.shape[0], "s and item_genres disagree on the number of films"
        self.register_buffer("log_s", torch.log(s + eps))                  # fixed data
        self.register_buffer("item_genres", item_genres)                   # fixed data
        self.alpha = nn.Parameter(torch.zeros(s.shape[-1]))                         # (T,)
        self.bias = nn.Parameter(torch.zeros(item_genres.shape[1], s.shape[-1]))    # (n_genres, T)

    def forward(self, users, items):
        """users, items: (B,) index tensors. Returns (B, T)."""
        profile = self.alpha * self.log_s[users, items]                    # (B, T)
        genre = self.item_genres[items] @ self.bias                        # (B, n_genres) @ (n_genres, T) -> (B, T)
        return profile + genre

    def full(self):
        """Every pair: (n_users, n_items, T)."""
        genre = self.item_genres @ self.bias                               # (n_items, T)
        return self.alpha * self.log_s + genre.unsqueeze(0)


class SumTerms(nn.Module):
    """Sum of several score terms (genre, year, ...), each with its own alpha and bias."""

    def __init__(self, terms):
        super().__init__()
        self.terms = nn.ModuleList(terms)        # registers their parameters and buffers

    def forward(self, users, items):
        """users, items: (B,). Returns (B, T)."""
        return sum(term(users, items) for term in self.terms)

    def full(self):
        """Every pair: (n_users, n_items, T)."""
        return sum(term.full() for term in self.terms)


class IMCGAE(nn.Module):
    """The full model: NodeEmbedding -> Encoder -> Decoder.

    Four independent options:
        features        named metadata pieces in the starting vectors, e.g.
                        combined genre/year and title embeddings
                        (dict {name: (N, n_k) matrix}, list of matrices, or None)
        share_feature_weights  use one learned projection per feature set for
                                every rating level, or separate projections
        layer_weights   one learned matrix per propagation layer
        side            rating-profile terms added to scores (GenreTerms, SumTerms or None)
        use_bias        learned level / user / item biases added to the scores
    """

    def __init__(self, n_users, n_items, T, d_shared=16, d_role=16, d_personal=16,
                 out_dim=64, n_layers=2, p0=0.1, theta=0.05, side=None, use_bias=False,
                 layer_weights=False, features=None, d_feat=8,
                 share_feature_weights=False):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.T = T
        self.embedding = NodeEmbedding(n_users, n_items, T, d_shared, d_role, d_personal,
                                       features, d_feat, share_feature_weights)
        self.encoder = Encoder(self.embedding.dim, out_dim, n_layers, p0, theta, layer_weights)
        self.decoder = Decoder(out_dim, T)
        self.side = side                                # GenreTerms, SumTerms or None

        # Optional learned biases on the scores, one number per rating level:
        #   level_bias[t]     how common level t is overall          (like mu in ALS)
        #   user_bias[u, t]   how user u uses level t                (like b_u)
        #   item_bias[i, t]   how film i receives level t            (like b_i)
        self.use_bias = use_bias
        if use_bias:
            self.level_bias = nn.Parameter(torch.zeros(T))                 # (T,)
            self.user_bias = nn.Parameter(torch.zeros(n_users, T))         # (n_users, T)
            self.item_bias = nn.Parameter(torch.zeros(n_items, T))         # (n_items, T)

    def describe(self):
        """What this model is made of, for printing and for saving next to the weights."""
        emb = self.embedding
        return {
            "starting_vector": {name: s.stop - s.start for name, s in emb.pieces().items()},
            "dim": emb.dim,
            "features": list(emb.feature_names),
            "share_feature_weights": emb.share_feature_weights,
            "n_layers": self.encoder.n_layers,
            "layer_weights": self.encoder.use_layer_weights,
            "encoder_dense_bias": self.encoder.dense.bias is not None,
            "side": self.side is not None,
            "use_bias": self.use_bias,
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }

    def encode(self, graphs):
        """Returns n: (N, out_dim) and h_t: (T, N, dim)."""
        return self.encoder(self.embedding(), graphs)

    def forward(self, graphs, users, items):
        """users, items: (B,) index tensors. Returns logits (B, T) and h_t (T, N, dim)."""
        n, h_t = self.encode(graphs)                    # (N, out_dim), (T, N, dim)
        n_u = n[users]                                  # (B, out_dim)
        n_i = n[self.n_users + items]                   # (B, out_dim)  items come after users
        logits = self.decoder(n_u, n_i)                 # (B, T)
        if self.side is not None:
            logits = logits + self.side(users, items)   # (B, T)
        if self.use_bias:
            logits = logits + self.level_bias + self.user_bias[users] + self.item_bias[items]   # (B, T)
        return logits, h_t                              # (B, T), (T, N, dim)

    @torch.no_grad()
    def predict_full(self, graphs, levels):
        """Predicted rating of every pair: (n_users, n_items). Dropout is off."""
        was_training = self.training
        self.eval()
        n, _ = self.encode(graphs)
        extra = self.side.full() if self.side is not None else None      # (n_users, n_items, T)
        if self.use_bias:
            bias = (self.level_bias                                      # (T,)
                    + self.user_bias[:, None, :]                         # (n_users, 1, T)
                    + self.item_bias[None, :, :])                        # (1, n_items, T)
            extra = bias if extra is None else extra + bias              # (n_users, n_items, T)
        table = self.decoder.predict_full(n[:self.n_users], n[self.n_users:], levels, extra=extra)
        self.train(was_training)
        return table
