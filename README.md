# Inductive Matrix Completion Using Graph Autoencoder for Sparse Rating Matrices

PyTorch implementation of **[Inductive Matrix Completion Using Graph Autoencoder](https://arxiv.org/abs/2108.11124) (Shen et al., CIKM 2021)** for matrix completion on a very sparse MovieLens subset (610 users × 4,980 movies, ~1% observed), with movie genres and release year as side information, compared with a biased ALS baseline.

Held-out test RMSE: tuned GNN 0.8682, biased ALS 0.8720, training-mean predictor 1.0369. See `report.pdf`.

## Layout
- `cf/`: data, rating graphs, features, IMC-GAE model, trainer (full-batch Adam, early stopping, cross-validation), ALS baseline
- `experiments/`: Optuna search and final fit (`notebook_imc_gae.ipynb`), baselines, plots


## Credits
R. Breaud, S. H. Nguyen, A. Aidoudi - Supervised by Alexandre Vérine.

Ratings from MovieLens (F. M. Harper and J. A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS 2015); not endorsed by GroupLens.
