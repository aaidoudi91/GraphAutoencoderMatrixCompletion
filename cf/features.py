import numpy as np


class GenreData:
    """Turns the (title, "Genre1|Genre2") array into a multi-hot matrix."""

    NO_GENRE = "(no genres listed)"

    def __init__(self, names_genres):
        # names_genres: (n_items, 2) array of strings, column 0 = title, column 1 = genres
        self.titles = names_genres[:, 0]
        self.n_items = len(names_genres)

        # One list of genres per film; a film without genre gets an empty list
        lists = [[] if g == self.NO_GENRE else g.split("|") for g in names_genres[:, 1]]

        self.genres = sorted({g for genres in lists for g in genres})   # genre names
        self.n_genres = len(self.genres)
        col = {g: k for k, g in enumerate(self.genres)}

        # multi_hot[i, k] = 1 if film i has genre k
        self.multi_hot = np.zeros((self.n_items, self.n_genres), dtype=np.float32)  # (n_items, n_genres)
        for i, genres in enumerate(lists):
            for g in genres:
                self.multi_hot[i, col[g]] = 1.0

    def normalised(self):
        """Each row divided by its number of genres, so every film has total weight 1 (or 0)."""
        count = self.multi_hot.sum(axis=1, keepdims=True)               # (n_items, 1)
        return self.multi_hot / np.maximum(count, 1.0)                  # (n_items, n_genres)

    def with_normalized_year(self, year_data):
        """Genre features with one standardized numeric release-year column appended."""
        year = np.asarray(year_data.normalized, dtype=np.float32).reshape(-1, 1)
        if year.shape[0] != self.n_items:
            raise ValueError(
                f"year data has {year.shape[0]} rows, expected {self.n_items}"
            )
        return np.concatenate((self.normalised(), year), axis=1)

    def of(self, i):
        """Genre names of film i, for inspection."""
        return [g for g, on in zip(self.genres, self.multi_hot[i]) if on]


def node_features(genres, n_users, n_items):
    """Genre matrix for all N graph nodes: (n_users + n_items, n_genres).

    Nodes are numbered rows of the rating matrix first, then columns
    (what the code calls "users" then "items"). The films are whichever
    side has as many entries as the genre file; the other side gets zeros.
    """
    feats = np.zeros((n_users + n_items, genres.n_genres), dtype=np.float32)
    on_rows = genres.n_items == n_users
    on_cols = genres.n_items == n_items
    if on_rows == on_cols:
        raise ValueError(
            f"cannot tell which side holds the films: genre file has {genres.n_items} "
            f"films, rating matrix is {n_users} x {n_items}")
    if on_rows:
        feats[:n_users] = genres.normalised()      # films are the first nodes
    else:
        feats[n_users:] = genres.normalised()      # films are the last nodes
    return feats



class GenreProfile:
    """How each user rates each genre, as a distribution over the rating levels.

    Built from training ratings only. For a pair (user u, film i):

        s[u, i, t] = mean over the genres g of film i of  P[u, g, t]

    P[u, g, t] is the share of u's ratings on genre-g films that are at level t,
    smoothed towards u's overall rating distribution (itself smoothed towards
    the global one). `shrink` is the number of "virtual" ratings of the prior.
    """

    def __init__(self, users, items, level_idx, multi_hot, n_users, T, shrink=10.0):
        self.users, self.items, self.level_idx = users, items, level_idx
        self.n_users, self.T, self.shrink = n_users, T, float(shrink)
        self.G = multi_hot.astype(np.float64)                          # (n_items, n_genres) 0/1
        count = self.G.sum(axis=1, keepdims=True)                      # (n_items, 1)
        self.Gn = self.G / np.maximum(count, 1.0)                      # rows sum to 1 (or 0)
        self.no_genre = count[:, 0] == 0                               # (n_items,)

        n_genres = self.G.shape[1]
        # C[u, g, t] = number of films of genre g that u rated at level t
        self.C = np.zeros((n_users, n_genres, T))
        for t in range(T):
            m = level_idx == t
            np.add.at(self.C[:, :, t], users[m], self.G[items[m]])
        # U[u, t] = number of ratings of u at level t
        self.U = np.zeros((n_users, T))
        np.add.at(self.U, (users, level_idx), 1.0)
        self.global_dist = np.bincount(level_idx, minlength=T) / len(level_idx)   # (T,)

    def _dists(self, C, U):
        """C: (..., n_genres, T), U: (..., T) -> q: (..., T), P: (..., n_genres, T)."""
        k = self.shrink
        q = (U + k * self.global_dist) / (U.sum(-1, keepdims=True) + k)            # user's overall distribution
        P = (C + k * q[..., None, :]) / (C.sum(-1, keepdims=True) + k)             # per genre
        return q, P

    def table(self):
        """s for every pair: (n_users, n_items, T), each [u, i] sums to 1.

        At the training pairs, the pair's own rating is removed from the counts
        first (leave-one-out), so the value never contains its own answer.
        """
        q, P = self._dists(self.C, self.U)                             # (n_users, T), (n_users, n_genres, T)
        s = np.einsum("ugt,ig->uit", P, self.Gn)                       # (n_users, n_items, T)
        s[:, self.no_genre, :] = q[:, None, :]                         # films without genre: overall distribution

        # leave-one-out at the training pairs
        u, i, lv = self.users, self.items, self.level_idx
        b = np.arange(len(u))
        C = self.C[u]                                                  # (B, n_genres, T) copy
        C[b, :, lv] -= self.G[i]                                       # remove the pair's own rating
        U = self.U[u]                                                  # (B, T) copy
        U[b, lv] -= 1.0
        q_b, P_b = self._dists(C, U)
        s_b = np.einsum("bgt,bg->bt", P_b, self.Gn[i])                 # (B, T)
        s_b[self.no_genre[i]] = q_b[self.no_genre[i]]
        s[u, i] = s_b
        return s.astype(np.float32)


class YearData:
    """Release years parsed from titles and standardized as one numeric feature.

    Missing years are filled with the median known year before standardizing,
    matching the continuous-year encoding used in the reference notebook.
    """

    def __init__(self, titles):
        import re
        self.years = np.zeros(len(titles), dtype=np.int64)             # 0 = unknown
        for i, title in enumerate(titles):
            found = re.findall(r"\((\d{4})\)", str(title))
            if found:
                self.years[i] = int(found[-1])                         # last "(dddd)" in the title

        known = self.years > 0
        self.n_missing = int((~known).sum())
        if known.any():
            known_years = self.years[known].astype(np.float32)
            self.median = float(np.median(known_years))
            filled = self.years.astype(np.float32)
            filled[~known] = self.median
            self.mean = float(filled.mean())
            self.scale = max(float(filled.std()), 1.0)
            self.normalized = ((filled - self.mean) / self.scale).astype(np.float32)
        else:
            self.median = self.mean = 0.0
            self.scale = 1.0
            self.normalized = np.zeros(len(titles), dtype=np.float32)


def on_film_nodes(item_matrix, n_users):
    """Put a per-film matrix (n_items, k) on the graph nodes: (n_users + n_items, k).

    Nodes are users first, then films, so the users' rows are zeros.
    """
    item_matrix = np.asarray(item_matrix, dtype=np.float32)
    out = np.zeros((n_users + item_matrix.shape[0], item_matrix.shape[1]), dtype=np.float32)
    out[n_users:] = item_matrix
    return out


DESCRIBE_PROMPT = ('In two sentences, describe the film "{title}": plot, tone and target audience. '
                   'Do not mention ratings, reviews or awards. '
                   'If you do not know this film, answer exactly: unknown')


def ask_ollama(prompt, model="llama3.1:8b", url="http://localhost:11434/api/generate", timeout=180):
    """Send one prompt to a model served locally by Ollama and return its answer."""
    import json
    import urllib.request
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0}}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.loads(reply.read())["response"].strip()


def describe_films(titles, cache_path, ask=ask_ollama, save_every=25, verbose=True):
    """One short description per film, written by a language model.

    Answers are stored in the JSON file `cache_path` as {title: text}, so the
    function can be stopped and run again: films already in the file are skipped.
    Returns the list of descriptions, in the order of `titles`.
    """
    import json
    from pathlib import Path
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def save():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))

    new = 0
    for k, title in enumerate(titles):
        title = str(title)
        if title in cache:
            continue
        cache[title] = ask(DESCRIBE_PROMPT.format(title=title)).strip()
        new += 1
        if verbose and new % save_every == 1:
            print(f"{k + 1}/{len(titles)}  {title} -> {cache[title][:90]}")
        if new % save_every == 0:
            save()
    save()
    return [cache[str(title)] for title in titles]


class TitleData:
    """Text features of the films: one vector per film, from a pretrained text model.

    texts   what is embedded for each film. By default "The movie <title without
            its year>". Pass the output of describe_films to embed descriptions
            instead. A film whose text is empty or starts with "unknown" is
            treated as unknown and gets a row of zeros everywhere.
    raw     (n_items, D) vectors from the text model, set by embed() or set_raw()
    """

    def __init__(self, titles, texts=None):
        import re
        self.titles = [str(t) for t in titles]
        self.n_items = len(self.titles)
        self.clean = [re.sub(r"\s*\(\d{4}\)\s*$", "", t) for t in self.titles]   # "Toy Story (1995)" -> "Toy Story"
        if texts is None:
            texts = [f"The movie {t}" for t in self.clean]
        assert len(texts) == self.n_items, "one text per film"
        self.texts = ["" if t is None else str(t).strip() for t in texts]
        self.known = np.array([bool(t) and not t.lower().startswith("unknown") for t in self.texts])
        self.raw = None

    def embed(self, model_name="all-MiniLM-L6-v2", cache_path=None):
        """Compute raw (n_items, D) with a sentence-transformers model.

        With cache_path, the vectors are loaded from that .npy file if it exists
        and has the right number of rows, and saved there otherwise.
        """
        from pathlib import Path
        if cache_path is not None and Path(cache_path).exists():
            raw = np.load(cache_path)
            if raw.shape[0] == self.n_items:
                return self.set_raw(raw)
        from sentence_transformers import SentenceTransformer         # imported here: only needed for this step
        encoder = SentenceTransformer(model_name)
        raw = np.asarray(encoder.encode(self.texts, show_progress_bar=True), dtype=np.float32)
        if cache_path is not None:
            np.save(cache_path, raw)
        return self.set_raw(raw)

    def set_raw(self, raw):
        """Use vectors computed elsewhere: (n_items, D). Unknown films are zeroed."""
        raw = np.asarray(raw, dtype=np.float32)
        assert raw.shape[0] == self.n_items, "one vector per film"
        self.raw = raw * self.known[:, None]
        return self

    def reduced(self, k=16):
        """PCA of the known films' vectors: (n_items, k), unknown films at zero.

        Every column is scaled so that a typical row has length about 1, the
        same scale as a normalised genre row.
        """
        assert self.raw is not None, "call embed() or set_raw() first"
        X = self.raw[self.known].astype(np.float64)                    # (n_known, D)
        X = X - X.mean(axis=0)
        k = min(k, X.shape[0], X.shape[1])
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        Z = X @ Vt[:k].T                                               # (n_known, k) main directions
        Z = Z / np.maximum(Z.std(axis=0), 1e-12) / np.sqrt(k)          # each column std 1/sqrt(k)
        out = np.zeros((self.n_items, k), dtype=np.float32)
        out[self.known] = Z
        return out

    def neighbours(self, query, k=5):
        """Titles of the k films whose vectors are closest to the first film matching `query`."""
        assert self.raw is not None, "call embed() or set_raw() first"
        i = query if isinstance(query, (int, np.integer)) else next(
            j for j, t in enumerate(self.titles) if str(query).lower() in t.lower())
        norm = np.linalg.norm(self.raw, axis=1)
        unit = self.raw / np.maximum(norm, 1e-12)[:, None]
        scores = unit @ unit[i]
        scores[i] = -np.inf
        scores[~self.known] = -np.inf
        return self.titles[i], [self.titles[j] for j in np.argsort(-scores)[:k]]
