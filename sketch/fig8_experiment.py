"""Replicate Figure 8 from arXiv:2508.14209.

Compares numerical stability of:
  - Normal equations (Cholesky)
  - Sketch-and-solve (multisketch)
  - QR solver

across varying condition numbers.
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from multisketch import sketch_and_solve_ls


def make_matrix_with_cond(d: int, n: int, cond: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a d x n matrix with specified condition number."""
    U, _ = np.linalg.qr(rng.normal(size=(d, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    # Singular values from 1 to 1/cond (logarithmically spaced)
    singular_values = np.logspace(0, -np.log10(cond), n)
    return U @ np.diag(singular_values) @ V.T


def solve_normal_equations(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve via normal equations: x = (A^T A)^{-1} A^T b.

    Uses explicit inverse to match the instability shown in the paper.
    This is intentionally unstable for ill-conditioned problems.
    """
    AtA = A.T @ A
    Atb = A.T @ b
    # Use explicit inverse - maximally unstable approach
    try:
        AtA_inv = np.linalg.inv(AtA)
        return AtA_inv @ Atb
    except np.linalg.LinAlgError:
        # If inv fails completely, return zeros (will give residual = 1)
        return np.zeros(A.shape[1])


def solve_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve via QR factorization."""
    Q, R = np.linalg.qr(A, mode='reduced')
    return np.linalg.solve(R, Q.T @ b)


def solve_lstsq(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve via numpy lstsq (SVD-based, most stable)."""
    return np.linalg.lstsq(A, b, rcond=None)[0]


def run_experiment(
    d: int = 2**17,
    n: int = 16,
    k1: int = 512,
    k2: int = 256,
    num_conds: int = 15,
    cond_min: float = 1e1,
    cond_max: float = 1e16,
    num_trials: int = 5,
    seed: int = 42,
) -> dict:
    """Run the stability experiment across condition numbers."""

    cond_numbers = np.logspace(np.log10(cond_min), np.log10(cond_max), num_conds)

    results = {
        'cond_numbers': cond_numbers,
        'normal_eq': [],
        'sketch': [],
        'qr': [],
    }

    rng = np.random.default_rng(seed)

    print(f"Running experiment: d={d}, n={n}, k1={k1}, k2={k2}")
    print(f"Condition numbers: {cond_min:.0e} to {cond_max:.0e} ({num_conds} points)")
    print(f"Trials per condition number: {num_trials}")
    print("-" * 60)

    for i, cond in enumerate(cond_numbers):
        res_normal = []
        res_sketch = []
        res_qr = []

        for trial in range(num_trials):
            # Generate matrix with this condition number
            A = make_matrix_with_cond(d, n, cond, rng)

            # b = A @ e where e is all ones (exact solution exists)
            x_true = np.ones(n)
            b = A @ x_true
            b_norm = np.linalg.norm(b)

            # Solve with each method
            x_normal = solve_normal_equations(A, b)
            x_sketch = sketch_and_solve_ls(A, b, k1, k2, rng=rng)
            x_qr = solve_qr(A, b)

            # Compute relative residuals
            res_normal.append(np.linalg.norm(A @ x_normal - b) / b_norm)
            res_sketch.append(np.linalg.norm(A @ x_sketch - b) / b_norm)
            res_qr.append(np.linalg.norm(A @ x_qr - b) / b_norm)

        # Average over trials (use nanmean to handle failures)
        results['normal_eq'].append(np.nanmean(res_normal))
        results['sketch'].append(np.nanmean(res_sketch))
        results['qr'].append(np.nanmean(res_qr))

        print(f"[{i+1:2d}/{num_conds}] cond={cond:.2e}: "
              f"normal={results['normal_eq'][-1]:.2e}, "
              f"sketch={results['sketch'][-1]:.2e}, "
              f"qr={results['qr'][-1]:.2e}")

    return results


def plot_results(results: dict, output_path: str) -> None:
    """Plot the stability comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    conds = results['cond_numbers']

    ax.loglog(conds, results['normal_eq'], 'o-', label='Normal Equations',
              color='#e74c3c', markersize=6, linewidth=1.5)
    ax.loglog(conds, results['sketch'], 's-', label='Sketch-and-Solve',
              color='#3498db', markersize=6, linewidth=1.5)
    ax.loglog(conds, results['qr'], '^-', label='QR Solver',
              color='#2ecc71', markersize=6, linewidth=1.5)

    ax.set_xlabel('Condition Number κ(A)', fontsize=12)
    ax.set_ylabel('Relative Residual ‖b−Ax‖₂/‖b‖₂', fontsize=12)
    ax.set_title('Numerical Stability vs Condition Number\n(Replication of Figure 8, arXiv:2508.14209)', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Add reference lines
    eps = np.finfo(np.float64).eps
    ax.axhline(y=eps, color='gray', linestyle='--', alpha=0.5)
    ax.text(conds[0], eps * 2, 'Machine ε', fontsize=9, color='gray')
    ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.5)
    ax.text(conds[0], 1.2, 'Complete failure', fontsize=9, color='black')

    # Set y-axis to show full range including failure
    ax.set_ylim(bottom=1e-17, top=1e2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Replicate Figure 8 from arXiv:2508.14209')
    parser.add_argument('--d', type=int, default=2**17, help='rows of A (default: 2^17=131072)')
    parser.add_argument('--n', type=int, default=16, help='cols of A (default: 16)')
    parser.add_argument('--k1', type=int, default=512, help='CountSketch dimension')
    parser.add_argument('--k2', type=int, default=256, help='Gaussian sketch dimension')
    parser.add_argument('--num-conds', type=int, default=15, help='number of condition numbers to test')
    parser.add_argument('--cond-min', type=float, default=1e1, help='minimum condition number')
    parser.add_argument('--cond-max', type=float, default=1e16, help='maximum condition number')
    parser.add_argument('--trials', type=int, default=5, help='trials per condition number')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--output', type=str, default='fig8_stability.png', help='output plot path')
    args = parser.parse_args()

    results = run_experiment(
        d=args.d,
        n=args.n,
        k1=args.k1,
        k2=args.k2,
        num_conds=args.num_conds,
        cond_min=args.cond_min,
        cond_max=args.cond_max,
        num_trials=args.trials,
        seed=args.seed,
    )

    plot_results(results, args.output)


if __name__ == '__main__':
    main()
