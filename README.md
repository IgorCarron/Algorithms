# AMP Compressed Sensing

Python implementation of Approximate Message Passing (AMP) for compressed sensing, based on:

**"Probabilistic reconstruction in compressed sensing: algorithms, phase diagrams, and threshold achieving matrices"**
F. Krzakala, M. Mézard, F. Sausset, Y. Sun, L. Zdeborová
Journal of Statistical Mechanics (2012) | [arXiv:1109.4424](https://arxiv.org/abs/1109.4424)

## Overview

AMP is a family of iterative algorithms for recovering sparse signals from underdetermined linear measurements. The key innovation from Krzakala et al. is using **Bayesian optimal denoisers** that exploit knowledge of the signal prior, achieving information-theoretically optimal reconstruction.

## Features

- **Multiple denoisers:**
  - Soft thresholding (L1/LASSO)
  - Gauss-Bernoulli Bayesian optimal (MMSE estimator)
  - Bernoulli denoiser

- **AMP with Onsager correction** for proper state evolution

- **Measurement matrices:**
  - Gaussian iid matrices
  - Spatially coupled (seeded) matrices for achieving information-theoretic limits

- **Experiments:**
  - Demo with visualization
  - Phase transition diagram generation

## Usage

```python
from amp_compressed_sensing import amp_reconstruction, generate_gaussian_matrix

# Generate problem
N = 1000  # signal dimension
M = 500   # measurements
rho = 0.1 # sparsity

A = generate_gaussian_matrix(M, N)
x_true = np.zeros(N)
support = np.random.choice(N, int(rho * N), replace=False)
x_true[support] = np.random.randn(len(support))
y = A @ x_true

# Reconstruct with Bayesian optimal denoiser
x_hat, history = amp_reconstruction(
    A, y,
    denoiser='gauss_bernoulli',
    rho=rho,
    verbose=True
)

print(f"MSE: {np.mean((x_hat - x_true)**2):.6f}")
```

## Demo

```bash
python amp_compressed_sensing.py
```

Generates:
- `amp_demo.png` - Reconstruction comparison (soft vs Bayesian)
- `phase_transition.png` - Success/failure phase diagram

## Results

With N=1000, rho=0.1 (10% sparsity), delta=0.5 (50% measurements):

| Method | MSE | Iterations |
|--------|-----|------------|
| Soft Thresholding | 0.000541 | 200 |
| Bayesian Optimal | 0.000040 | 51 |

The Bayesian denoiser achieves ~13x better MSE, confirming the paper's theoretical predictions.

## Phase Transition

The algorithm exhibits a sharp phase transition in the (delta, rho) plane:
- **Green region** (high delta, low rho): Perfect recovery
- **Red region** (low delta, high rho): Recovery fails
- Transition occurs near delta ≈ 2*rho for Gaussian matrices

## Requirements

- NumPy
- Matplotlib (for visualization)

## References

- Krzakala, Mézard, Sausset, Sun, Zdeborová. ["Probabilistic reconstruction in compressed sensing"](https://arxiv.org/abs/1109.4424). JSTAT 2012.
- Donoho, Maleki, Montanari. ["Message passing algorithms for compressed sensing"](https://arxiv.org/abs/0907.3574). PNAS 2009.
