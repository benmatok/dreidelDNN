
import torch
import torch.nn as nn
import numpy as np
import scipy.linalg

def solve_diagonal(X, Y, P_indices=None):
    # Y ~ FWHT(P(X) * D)
    # Y = H * (X_perm * D) / N
    # Y * N = H * (X_perm * D)
    # H^T * Y * N = N * (X_perm * D)
    # H^T * Y = X_perm * D
    # Let Z = H^T * Y.
    # Z_j = X_perm_j * D_j
    # D_j = <X_perm_j, Z_j> / <X_perm_j, X_perm_j>

    N = X.shape[1]
    if P_indices is not None:
        X_perm = X[:, P_indices]
    else:
        X_perm = X

    H = scipy.linalg.hadamard(N).astype(np.float32)
    Z = np.dot(Y, H) # Y is (B, N). H is symmetric.

    numerator = np.sum(X_perm * Z, axis=0)
    denominator = np.sum(X_perm**2, axis=0)

    D = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=1e-9)
    return D

def test_approximation():
    N = 128
    B = 128

    # Random Dense Matrix W
    W = np.random.randn(N, N) / np.sqrt(N)

    X = np.random.randn(B, N)
    Y_target = np.dot(X, W)

    # 1. K=1 (Original LinearWHT)
    D1 = solve_diagonal(X, Y_target)

    H = scipy.linalg.hadamard(N).astype(np.float32)
    Y_pred1 = np.dot(X * D1, H) / N

    rel_err1 = np.linalg.norm(Y_pred1 - Y_target) / np.linalg.norm(Y_target)
    print(f"K=1 Rel Error: {rel_err1:.4f}")

    # 2. K=4 (Greedy Residual)
    Y_curr = Y_target.copy()
    Y_accum = np.zeros_like(Y_target)

    branches = []

    for k in range(8):
        # Permutation
        if k == 0:
            perm = np.arange(N)
        else:
            perm = np.random.permutation(N)

        D = solve_diagonal(X, Y_curr, perm)

        # Predict this branch
        X_perm = X[:, perm]
        Y_branch = np.dot(X_perm * D, H) / N

        Y_accum += Y_branch
        Y_curr = Y_target - Y_accum

        branches.append((perm, D))

        rel_err_k = np.linalg.norm(Y_curr) / np.linalg.norm(Y_target)
        print(f"K={k+1} Rel Error: {rel_err_k:.4f}")

if __name__ == "__main__":
    test_approximation()
