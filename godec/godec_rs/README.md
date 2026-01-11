# Rust GoDec

This folder contains a Rust implementation of GoDec exposed as the `godec_rs` Python module.
The top-level `godec.py` in the repo will prefer the Rust module when it is installed.

Build the Python extension (requires Rust and maturin):

```
cd godec_rs
maturin develop --release
```

Python usage mirrors the original:

```
from godec import godec
L, S, LS, RMSE = godec(X, rank=1, card=None, iterated_power=1, max_iter=100, tol=0.001)
```

You can also choose a random projection:

```
L, S, LS, RMSE = godec(X, projection="srht")
```

Options: `gaussian`, `rademacher`, `achlioptas`, `sparse_jl`, `srht`, `srft`, `block_sparse`.
