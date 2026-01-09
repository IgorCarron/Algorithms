"""CPU multisketch least-squares solver.

Implements the sketch-and-solve pipeline using CountSketch followed by
Gaussian sketching as described in arXiv:2508.14209.
"""
from __future__ import annotations

import argparse
import time
import numpy as np


def countsketch_apply(A: np.ndarray, r: np.ndarray, s: np.ndarray, k: int) -> np.ndarray:
    """Apply CountSketch without forming the sketch matrix.

    A: (d, n) matrix or (d,) vector
    r: length-d indices in [0, k)
    s: length-d signs in {-1, +1}
    k: sketch dimension
    """
    A_arr = np.asarray(A)
    if A_arr.ndim == 1:
        y = np.zeros(k, dtype=A_arr.dtype)
        np.add.at(y, r, s * A_arr)
        return y
    if A_arr.ndim == 2:
        y = np.zeros((k, A_arr.shape[1]), dtype=A_arr.dtype)
        np.add.at(y, r, s[:, None] * A_arr)
        return y
    raise ValueError("A must be 1D or 2D")


def gaussian_sketch_apply(A: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply a Gaussian sketch with 1/sqrt(k) scaling."""
    A_arr = np.asarray(A)
    G = rng.normal(0.0, 1.0 / np.sqrt(k), size=(k, A_arr.shape[0]))
    return G @ A_arr, G


def sketch_and_solve_ls(
    A: np.ndarray,
    b: np.ndarray,
    k1: int,
    k2: int,
    rng: np.random.Generator | None = None,
    return_sketch: bool = False,
) -> tuple[np.ndarray, dict] | np.ndarray:
    """Solve min_x ||Ax - b||_2 with CountSketch + Gaussian (multisketch)."""
    A = np.asarray(A)
    b = np.asarray(b)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    d, n = A.shape
    if b.shape[0] != d:
        raise ValueError("b length must match A rows")
    if rng is None:
        rng = np.random.default_rng()

    r = rng.integers(0, k1, size=d)
    s = rng.choice(np.array([-1, 1], dtype=A.dtype), size=d)

    A1 = countsketch_apply(A, r, s, k1)
    b1 = countsketch_apply(b, r, s, k1)

    A2, G = gaussian_sketch_apply(A1, k2, rng)
    b2 = G @ b1

    Q, R = np.linalg.qr(A2, mode="reduced")
    x = np.linalg.solve(R, Q.T @ b2)

    if return_sketch:
        return x, {"r": r, "s": s, "A1": A1, "b1": b1, "G": G}
    return x


def exact_ls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference least-squares solution via dense solver."""
    return np.linalg.lstsq(A, b, rcond=None)[0]


def run_demo(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    d = args.d
    n = args.n
    k1 = args.k1
    k2 = args.k2

    A = rng.normal(size=(d, n))
    x_true = rng.normal(size=n)
    b = A @ x_true
    if args.noise > 0:
        b = b + args.noise * rng.normal(size=d)

    x_sketch = sketch_and_solve_ls(A, b, k1, k2, rng=rng)
    x_exact = exact_ls(A, b)

    res_sketch = np.linalg.norm(A @ x_sketch - b) / np.linalg.norm(b)
    res_exact = np.linalg.norm(A @ x_exact - b) / np.linalg.norm(b)
    err_sketch = np.linalg.norm(x_sketch - x_true) / np.linalg.norm(x_true)
    err_exact = np.linalg.norm(x_exact - x_true) / np.linalg.norm(x_true)

    print("multisketch residual rel:", res_sketch)
    print("exact residual rel:", res_exact)
    print("multisketch x error rel:", err_sketch)
    print("exact x error rel:", err_exact)

def _parse_int_list(value: str) -> list[int]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise ValueError("list cannot be empty")
    return [int(v) for v in items]


def run_sweep(args: argparse.Namespace) -> None:
    rng_data = np.random.default_rng(args.seed)
    rng_sketch = np.random.default_rng(args.seed + 1)

    d = args.d
    n = args.n
    k1_list = _parse_int_list(args.k1_list)
    k2_list = _parse_int_list(args.k2_list)

    A = rng_data.normal(size=(d, n))
    x_true = rng_data.normal(size=n)
    b = A @ x_true
    if args.noise > 0:
        b = b + args.noise * rng_data.normal(size=d)

    x_exact = exact_ls(A, b)
    res_exact = np.linalg.norm(A @ x_exact - b) / np.linalg.norm(b)
    err_exact = np.linalg.norm(x_exact - x_true) / np.linalg.norm(x_true)
    print("exact residual rel:", res_exact)
    print("exact x error rel:", err_exact)
    print("")
    print("k1,k2,avg_ms,residual_rel,x_error_rel")

    results = []
    for k1 in k1_list:
        for k2 in k2_list:
            times = []
            res_list = []
            err_list = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                x_sketch = sketch_and_solve_ls(A, b, k1, k2, rng=rng_sketch)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
                res_list.append(np.linalg.norm(A @ x_sketch - b) / np.linalg.norm(b))
                err_list.append(np.linalg.norm(x_sketch - x_true) / np.linalg.norm(x_true))
            avg_ms = float(np.mean(times))
            avg_res = float(np.mean(res_list))
            avg_err = float(np.mean(err_list))
            results.append((k1, k2, avg_ms, avg_res, avg_err))
            print(f"{k1},{k2},{avg_ms:.3f},{avg_res:.6e},{avg_err:.6e}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - optional dependency
            print("plot requested but matplotlib is unavailable:", exc)
            return
        k1_vals = [r[0] for r in results]
        k2_vals = [r[1] for r in results]
        times = [r[2] for r in results]
        errors = [r[4] for r in results]
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        scatter = ax.scatter(times, errors, c=k2_vals, s=45, cmap="viridis")
        ax.set_xlabel("avg ms")
        ax.set_ylabel("x error rel")
        ax.set_title("Multisketch sweep")
        cb = fig.colorbar(scatter, ax=ax)
        cb.set_label("k2")
        for k1, k2, t, _, e in results:
            ax.annotate(f"k1={k1}", (t, e), fontsize=7, alpha=0.7)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=160)
        print("saved plot:", args.plot)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU multisketch least-squares demo")
    parser.add_argument("--d", type=int, default=10000, help="rows of A")
    parser.add_argument("--n", type=int, default=50, help="cols of A")
    parser.add_argument("--k1", type=int, default=400, help="CountSketch dimension")
    parser.add_argument("--k2", type=int, default=200, help="Gaussian sketch dimension")
    parser.add_argument("--k1-list", type=str, default="200,400,800", help="comma-separated k1 values for sweep")
    parser.add_argument("--k2-list", type=str, default="100,200,400", help="comma-separated k2 values for sweep")
    parser.add_argument("--repeats", type=int, default=3, help="repeats per (k1,k2) in sweep")
    parser.add_argument("--noise", type=float, default=0.0, help="noise std for b")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--sweep", action="store_true", help="run a timing/accuracy sweep")
    parser.add_argument("--plot", type=str, default="", help="save a small plot to this path")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.sweep:
        run_sweep(args)
    else:
        run_demo(args)
