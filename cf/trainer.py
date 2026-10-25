import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from cf.graph import RatingGraphs
from cf.models.imc_gae import Decoder


class Trainer:
    """Trains an IMCGAE model with full-batch Adam (or AdamW) and early stopping.

    graphs must be built from the TRAINING ratings only.
    levels is the array of rating values, e.g. [1, 2, 3, 4, 5].

    optimizer:
        "adam"   (default) weight decay is added to the gradient (L2). With this
                 model the data gradient of the personal vectors is tiny, so
                 Adam turns the decay into "move one lr towards zero per step"
                 and wears them down. That acts as a strong regulariser, and
                 on this data it gives the best validation RMSE.
        "adamw"  weight decay is applied apart from the gradient: every step
                 multiplies the weights by (1 - lr * weight_decay). The personal
                 vectors survive and the model memorises quickly (validation
                 0.92 against 0.89 in our check), unless weight_decay is large.
    """

    def __init__(self, model, graphs, levels, lr=0.01, lam=0.0, epochs=1000,
                 eval_every=5, patience=20, verbose=True, weight_decay=1e-2,
                 optimizer="adam"):
        self.model = model
        self.graphs = graphs
        self.device = graphs.device
        self.model.to(self.device)
        self.levels = torch.as_tensor(levels, dtype=torch.float32, device=self.device)   # (T,)
        self.lam = lam                # weight of the cosine term
        self.epochs = epochs          # maximum number of training steps
        self.eval_every = eval_every  # validate every this many steps
        self.patience = patience      # stop after this many validations without progress
        self.verbose = verbose
        self.optimizer = self._make_optimizer(model, lr, weight_decay, optimizer)
        self.history = []

    @staticmethod
    def _make_optimizer(model, lr, weight_decay, kind):
        """Two parameter groups: everything with weight decay, the score terms
        of `model.side` (a handful of numbers) without."""
        decay = [p for n, p in model.named_parameters() if not n.startswith("side.")]
        no_decay = [p for n, p in model.named_parameters() if n.startswith("side.")]
        groups = [{"params": decay, "weight_decay": weight_decay}]
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        if kind == "adamw":
            return torch.optim.AdamW(groups, lr=lr)
        if kind == "adam":
            return torch.optim.Adam(groups, lr=lr)
        raise ValueError(f"unknown optimizer {kind!r}, use 'adamw' or 'adam'")

    def _tensors(self, triples):
        users, items, level_idx, ratings = triples
        as_long = lambda a: torch.as_tensor(a, dtype=torch.long, device=self.device)
        return (as_long(users), as_long(items), as_long(level_idx),
                torch.as_tensor(ratings, dtype=torch.float32, device=self.device))

    def loss(self, logits, targets, h_t):
        """Cross-entropy minus lam * mean similarity of adjacent rating levels.

        logits: (B, T), targets: (B,) level indices, h_t: (T, N, dim).
        Returns the total loss and its two parts.
        """
        ce = F.cross_entropy(logits, targets)                             # scalar, mean over the B pairs
        if h_t.shape[0] > 1:
            cos = F.cosine_similarity(h_t[:-1], h_t[1:], dim=-1)          # (T-1, N)  level t vs t+1
            reg = -cos.mean()                                             # scalar, normalized by transitions and nodes
        else:
            reg = torch.zeros((), device=logits.device)
        return ce + self.lam * reg, ce, reg

    @torch.no_grad()
    def evaluate(self, triples):
        """RMSE on a set of (users, items, level_idx, ratings), with dropout off."""
        users, items, _, ratings = self._tensors(triples)
        self.model.eval()
        logits, _ = self.model(self.graphs, users, items)                 # (B, T)
        pred = Decoder.expected_rating(logits, self.levels)               # (B,)
        return torch.sqrt(torch.mean((pred - ratings) ** 2)).item()

    @torch.no_grad()
    def evaluate_all(self, triples):
        """RMSE and the loss with its two parts on a set of ratings, with dropout off."""
        users, items, targets, ratings = self._tensors(triples)
        self.model.eval()
        logits, h_t = self.model(self.graphs, users, items)               # (B, T), (T, N, dim)
        loss, ce, reg = self.loss(logits, targets, h_t)
        pred = Decoder.expected_rating(logits, self.levels)               # (B,)
        return {"rmse": torch.sqrt(torch.mean((pred - ratings) ** 2)).item(),
                "ce": ce.item(),                                          # log loss on these ratings
                "cos_loss": self.lam * reg.item(),                        # weighted mean cosine penalty
                "cos_mean": -reg.item(),                                 # mean cosine similarity
                "loss": loss.item()}                                      # ce + cos_loss

    def fit(self, train, val=None):
        """Trains on `train`. If `val` is given, keeps the weights with the best
        validation RMSE and stops early. Returns the history (a list of dicts)."""
        users, items, targets, _ = self._tensors(train)
        best_rmse, best_state, bad_rounds = float("inf"), None, 0
        start = time.time()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            logits, h_t = self.model(self.graphs, users, items)           # forward
            loss, ce, reg = self.loss(logits, targets, h_t)
            loss.backward()                                               # gradients
            self.optimizer.step()                                         # update

            if epoch % self.eval_every == 0 or epoch == self.epochs:
                tr = self.evaluate_all(train)
                row = {"epoch": epoch,
                       "train_loss": tr["loss"], "train_ce": tr["ce"],
                       "cos_loss": tr["cos_loss"],
                       "cos_mean": tr["cos_mean"],
                       "train_rmse": tr["rmse"],
                       "time": time.time() - start}
                row.update(self.diagnostics(train))
                if val is not None:
                    va = self.evaluate_all(val)
                    row["val_loss"] = va["loss"]
                    row["val_ce"] = va["ce"]
                    row["val_rmse"] = va["rmse"]
                    if row["val_rmse"] < best_rmse - 1e-5:
                        best_rmse, bad_rounds = row["val_rmse"], 0
                        best_state = copy.deepcopy(self.model.state_dict())
                    else:
                        bad_rounds += 1
                self.history.append(row)
                if self.verbose:
                    print("  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                    for k, v in row.items()))
                if val is not None and bad_rounds >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    @classmethod
    def cross_validate(cls, data, model_factory, side_factory=None, params=None,
                       n_splits=5, seed=0, epochs=300, eval_every=5,
                       patience=10, verbose=False):
        """Run rating-level K-fold validation and return fold metrics.

        ``model_factory(params, side)`` must create a fresh model. If supplied,
        ``side_factory(train, params)`` is called separately for every fold with only
        that fold's training quadruple (users, items, level indices, ratings),
        so profile features can be built without validation leakage.

        For every fold, ratings that would leave a user or item with no training
        observations are kept in that fold's training set. Consequently, fold
        validation sizes can be slightly smaller than an exact 1 / n_splits.
        The data object is expected to expose the RatingData attributes.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        n_ratings = len(data.ratings)
        if n_splits > n_ratings:
            raise ValueError("n_splits cannot exceed the number of observed ratings")
        params = {} if params is None else dict(params)

        rng = np.random.default_rng(seed)
        folds = np.array_split(rng.permutation(n_ratings), n_splits)
        fold_rmse, fold_train_rmse, best_epochs, fold_sizes = [], [], [], []

        for fold_number, candidate_val_idx in enumerate(folds):
            train_mask = np.ones(n_ratings, dtype=bool)
            train_mask[candidate_val_idx] = False
            train_user_count = np.bincount(
                data.users[train_mask], minlength=data.n_users
            )
            train_item_count = np.bincount(
                data.items[train_mask], minlength=data.n_items
            )

            # Preserve at least one train edge for every represented user/item.
            for idx in candidate_val_idx:
                user, item = data.users[idx], data.items[idx]
                if train_user_count[user] == 0 or train_item_count[item] == 0:
                    train_mask[idx] = True
                    train_user_count[user] += 1
                    train_item_count[item] += 1

            train_idx = np.flatnonzero(train_mask)
            val_idx = np.flatnonzero(~train_mask)
            if val_idx.size == 0:
                raise ValueError(f"fold {fold_number} has no validation ratings")

            train = (data.users[train_idx], data.items[train_idx],
                     data.level_idx[train_idx], data.ratings[train_idx])
            val = (data.users[val_idx], data.items[val_idx],
                   data.level_idx[val_idx], data.ratings[val_idx])

            # Keep initialization and dropout reproducible across parameter trials.
            torch.manual_seed(seed + fold_number)
            np.random.seed(seed + fold_number)
            graphs = RatingGraphs(train[0], train[1], train[2],
                                  data.n_users, data.n_items, data.T)
            side = side_factory(train, params) if side_factory is not None else None
            model = model_factory(params, side)
            trainer = cls(
                model, graphs, data.levels,
                lr=params.get("lr", 0.01),
                lam=params.get("lam", 0.0),
                weight_decay=params.get("weight_decay", 1e-2),
                optimizer=params.get("optimizer", "adam"),
                epochs=epochs, eval_every=eval_every,
                patience=patience, verbose=verbose,
            )
            history = trainer.fit(train, val)
            best = min(history, key=lambda row: row["val_rmse"])
            fold_rmse.append(float(best["val_rmse"]))
            fold_train_rmse.append(float(best["train_rmse"]))
            best_epochs.append(int(best["epoch"]))
            fold_sizes.append({"train": len(train_idx), "val": len(val_idx)})

        return {
            "mean_rmse": float(np.mean(fold_rmse)),
            "std_rmse": float(np.std(fold_rmse)),
            "fold_rmse": fold_rmse,
            "fold_train_rmse": fold_train_rmse,     # train RMSE at each fold's best epoch
            "best_epochs": best_epochs,
            "fold_sizes": fold_sizes,
        }

    @torch.no_grad()
    def diagnostics(self, triples):
        """One number per weight group, plus the size of the scores."""
        m = self.model
        rms = lambda p: p.pow(2).mean().sqrt().item()
        users, items, _, _ = self._tensors(triples)
        m.eval()
        logits, _ = m(self.graphs, users, items)
        probs = torch.softmax(logits, dim=-1)
        out = {"w_shared": rms(m.embedding.shared),
               "w_role": rms(m.embedding.role.weight),
               "w_personal": rms(m.embedding.personal)}
        feat = getattr(m.embedding, "feat", [])                           # named metadata projection weights
        names = getattr(m.embedding, "feature_names", [f"feat{k}" for k in range(len(feat))])
        for name, w in zip(names, feat):
            out[f"w_{name}"] = rms(w)
        out.update({"w_dense": rms(m.encoder.dense.weight),
                    "w_decoder": rms(m.decoder.weight),
                    "logit_std": logits.std().item(),
                    "confidence": probs.max(dim=-1).values.mean().item()})
        return out

    @torch.no_grad()
    def ablation(self, triples, feature_names=None):
        """RMSE when one piece of the model is zeroed; the weights are restored afterwards.

        feature_names: names of the feature sets, in the model's order. By
        default the names stored in the model are used.
        """
        model, emb = self.model, self.model.embedding
        feat = getattr(emb, "feat", [])
        if feature_names is None:
            feature_names = getattr(emb, "feature_names", [f"feat{k}" for k in range(len(feat))])
        assert len(feature_names) == len(feat), "one name per feature set"

        pieces = [("no_shared",   [emb.shared]),
                  ("no_role",     [emb.role.weight]),
                  ("no_personal", [emb.personal])]

        # node-feature projection groups, e.g. combined genre/year or title
        for name, w in zip(feature_names, feat):
            pieces.append((f"no_{name}", [w]))

        # terms added to the scores, if the model has them
        if getattr(model, "side", None) is not None:
            pieces.append(("no_side", list(model.side.parameters())))
        if getattr(model, "use_bias", False):
            pieces.append(("no_bias", [model.level_bias, model.user_bias, model.item_bias]))

        out = {"full": self.evaluate(triples)}
        for name, params in pieces:
            saved = [p.detach().clone() for p in params]
            for p in params:
                p.zero_()
            out[name] = self.evaluate(triples)
            for p, s in zip(params, saved):
                p.copy_(s)
        return out
