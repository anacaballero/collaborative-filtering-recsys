# Collaborative Filtering Recommender System

Recommender system built on matrix factorization (FunkSVD) with SGD-based latent factor learning, cold-start handling, and a complete evaluation framework.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`matrix_factorization_funksvd.ipynb`](matrix_factorization_funksvd.ipynb) | FunkSVD implemented from scratch in NumPy. Updates user and item latent factor matrices via SGD on observed ratings only, avoiding imputation bias. Includes convergence analysis. |
| [`train_val_test_split_collaborative.ipynb`](train_val_test_split_collaborative.ipynb) | User-stratified train/validation/test splits that preserve per-user rating distribution. Prevents data leakage in sparse matrix settings. |
| [`recommender_evaluation_funksvd.ipynb`](recommender_evaluation_funksvd.ipynb) | Evaluation of trained FunkSVD on held-out user-item pairs. Computes RMSE, analyzes error distribution, identifies user/item segments where the model underperforms. |
| [`cold_start_recommendation.ipynb`](cold_start_recommendation.ipynb) | Cold-start strategy using a composite ranking score (average rating + rating volume + recency) with a minimum-interaction threshold to filter noise. |

## Key Concepts

**FunkSVD update rule** — SGD on observed ratings only:
```
for (i, j) where R[i,j] > 0:
    diff = R[i,j] - U[i,:] · V[:,j]
    U[i,:] += lr * 2 * diff * V[:,j]
    V[:,j] += lr * 2 * diff * U[i,:]
```

**Cold-start ranking** — for new users with no interaction history, items are ranked by:
- Average rating (weighted by number of ratings)
- Rating volume (popularity signal)
- Recency (time-decay weighting)

## Stack
- NumPy, Pandas
- scikit-learn (evaluation utilities)
- MovieLens dataset
