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
