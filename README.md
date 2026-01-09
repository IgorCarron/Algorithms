# Binary Matrix Factorization

Python implementation of the algorithms from:

**"Matrix Factorization with Binary Components"**
Martin Slawski, Matthias Hein, Pavlo Lutsik
NeurIPS 2013 | [arXiv:1401.6024](https://arxiv.org/abs/1401.6024)

## Problem

Given a data matrix **D** ∈ ℝ^(m×n), find:
- **T** ∈ {0,1}^(m×r) — binary factor matrix
- **A** ∈ ℝ^(r×n) — coefficient matrix

such that **D = T·A**

Optional constraint: columns of **A** sum to 1 (convex combinations).

## Key Features

- **Exact factorization** (Algorithms 1 & 2): Finds all binary vertices in aff(D) using the Littlewood-Offord lemma
- **Approximate factorization** (Algorithm 3): Handles noisy data via SVD-based initialization
- **Block optimization** (Algorithm 4): Alternating refinement with optional non-negativity and simplex constraints
- **O(m·2^(r-1))** complexity — tractable for small rank r

## Applications

- DNA methylation unmixing (cell type deconvolution)
- Binary classification ensembles
- Topic modeling with hard assignments

## Usage

```python
from binary_matrix_factorization import binary_factorization_exact, binary_factorization_approximate

# Exact factorization (noiseless data)
T, A = binary_factorization_exact(D, affine=True, verbose=True)

# Approximate factorization (noisy data)
T, A = binary_factorization_approximate(D, r=4, nonnegative_A=True, sum_to_one=True)
```

## Demo

```bash
python binary_matrix_factorization.py
```

Runs demos for:
1. Exact factorization on synthetic data
2. Approximate factorization with noise
3. Separable case (unique solution)
4. DNA methylation unmixing simulation

## Requirements

- NumPy
- SciPy

## References

- Slawski, Hein, Lutsik. ["Matrix Factorization with Binary Components"](https://arxiv.org/abs/1401.6024). NeurIPS 2013.
- Lutsik et al. ["MeDeCom: Discovery and quantification of latent components of heterogeneous methylomes"](https://doi.org/10.1186/s12859-019-2865-6). BMC Bioinformatics 2017.
