# Algorithms

A collection of algorithm implementations from recent research papers.

## KOBO - Kernel Optimized Black-Box Optimization

Implementation of the KOBO algorithm from:

**"Kernel Learning for Sample Constrained Black-Box Optimization"**
Rajagopalan, Wei, Roy Choudhury (AAAI 2025)
[arXiv:2507.20533](https://arxiv.org/abs/2507.20533)

The algorithm learns an optimal GPR kernel for black-box optimization by:
1. Creating composite kernels via a context-free grammar (Kernel Combiner)
2. Learning a continuous latent space of kernels (KerVAE)
3. Optimizing model evidence in this latent space (KerGPR)
4. Using the optimal kernel for black-box optimization (fGPR)

**Files:**
- `kobo/kobo.py` - Main KOBO implementation
- `kobo/kobo_improved.py` - Improved variant
- `kobo/kobo_comparison.png` - Performance comparison

## Multisketch Least Squares

Implementation of the sketch-and-solve pipeline from [arXiv:2508.14209](https://arxiv.org/abs/2508.14209) using CountSketch followed by a Gaussian sketch.

**Files:**
- `sketch/multisketch.py` - Main implementation and CLI demo

**Usage:**
```bash
python sketch/multisketch.py --d 10000 --n 50 --k1 400 --k2 200
```

## Binary Matrix Factorization

Implementation of the algorithms from:

**"Matrix Factorization with Binary Components"**
Slawski, Hein, Lutsik (NeurIPS 2013)
[arXiv:1401.6024](https://arxiv.org/abs/1401.6024)

Given a data matrix **D**, find binary factor matrix **T** ∈ {0,1} and coefficient matrix **A** such that **D = T·A**.

**Features:**
- Exact factorization using the Littlewood-Offord lemma
- Approximate factorization for noisy data via SVD initialization
- Block optimization with optional non-negativity and simplex constraints

**Files:**
- `binary-matrix-factorization/binary_matrix_factorization.py` - Main implementation

**Usage:**
```python
from binary_matrix_factorization import binary_factorization_exact, binary_factorization_approximate

T, A = binary_factorization_exact(D, affine=True)
T, A = binary_factorization_approximate(D, r=4, nonnegative_A=True, sum_to_one=True)
```

## CKS - Compositional Kernel Search

Implementation of the Compositional Kernel Search algorithm from:

**"Structure Discovery in Nonparametric Regression through Compositional Kernel Search"**
Duvenaud, Lloyd, Grosse, Tenenbaum, Ghahramani (ICML 2013)
[arXiv:1302.4922](https://arxiv.org/abs/1302.4922)

Automatic structure discovery in time series and regression data using Gaussian Processes with compositional kernels.

**Files:**
- `cks/compositional_kernel_search.py` - Main CKS implementation
- `cks/examples.py` - Example usage and demonstrations

## AMP Compressed Sensing

Implementation of Approximate Message Passing for compressed sensing from:

**"Probabilistic reconstruction in compressed sensing: algorithms, phase diagrams, and threshold achieving matrices"**
Krzakala, Mézard, Sausset, Sun, Zdeborová (JSTAT 2012)
[arXiv:1109.4424](https://arxiv.org/abs/1109.4424)

Recovers sparse signals from underdetermined linear measurements using Bayesian optimal denoisers.

**Features:**
- Multiple denoisers (soft thresholding, Gauss-Bernoulli, Bernoulli)
- AMP with Onsager correction
- Gaussian and spatially coupled measurement matrices

**Files:**
- `amp-compressed-sensing/amp_compressed_sensing.py` - Main implementation

**Usage:**
```python
from amp_compressed_sensing import amp_reconstruction, generate_gaussian_matrix

A = generate_gaussian_matrix(M, N)
x_hat, history = amp_reconstruction(A, y, denoiser='gauss_bernoulli', rho=0.1)
```


## Implementations

| Paper | ArXiv | Authors |
| --- | --- | --- |
| [Sparser Johnson-Lindenstrauss Transforms](./Sparse_Johnson_Lindenstrauss_arxiv_1012_1577) | [1012.1577](https://arxiv.org/abs/1012.1577) | Daniel M. Kane, Jelani Nelson |
