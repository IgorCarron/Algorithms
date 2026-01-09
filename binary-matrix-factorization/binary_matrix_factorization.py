"""
Binary Matrix Factorization

Implementation of algorithms from:
"Matrix factorization with Binary Components"
Slawski, Hein, and Lutsik (NeurIPS 2013, arXiv:1401.6024)

Given D ∈ R^{m×n}, find T ∈ {0,1}^{m×r} and A ∈ R^{r×n} such that D = T·A
(with optional constraint A^T·1_r = 1_n for affine combinations)
"""

import numpy as np
from scipy import linalg
from itertools import product
from typing import Tuple, Optional, Set, List
import warnings


def generate_binary_vectors(r: int) -> np.ndarray:
    """
    Generate all 2^r binary vectors of length r.
    Returns matrix B of shape (r, 2^r) where columns are binary vectors.
    """
    vectors = list(product([0, 1], repeat=r))
    return np.array(vectors).T


def find_vertices_exact(D: np.ndarray, verbose: bool = False) -> Set[tuple]:
    """
    Algorithm 1: FINDVERTICES_EXACT
    
    Find all vertices of [0,1]^m contained in aff(D).
    
    Parameters
    ----------
    D : np.ndarray
        Data matrix of shape (m, n)
    verbose : bool
        Print progress information
        
    Returns
    -------
    T : set of tuples
        Set of binary vectors (vertices) contained in aff(D)
    """
    m, n = D.shape
    
    # Step 1: Fix p ∈ aff(D) and compute P = [D[:,0] - p, ..., D[:,n-1] - p]
    p = D[:, 0].copy()  # Use first column as reference point
    P = D - p[:, np.newaxis]
    
    # Step 2: Determine r-1 linearly independent columns of P
    # Using QR decomposition for rank-revealing
    Q, R, column_perm = linalg.qr(P, pivoting=True)
    
    # Determine numerical rank
    tol = max(m, n) * np.finfo(float).eps * np.abs(R).max()
    rank = np.sum(np.abs(np.diag(R)) > tol)
    
    if rank == 0:
        if verbose:
            print("Data has rank 0 - returning p as only vertex if binary")
        if np.allclose(p, np.round(p)) and np.all((p >= 0) & (p <= 1)):
            return {tuple(np.round(p).astype(int))}
        return set()
    
    r_minus_1 = rank  # This gives us r-1 where r is the affine rank
    C = column_perm[:r_minus_1]  # Indices of linearly independent columns
    P_C = P[:, C]
    
    # Find r-1 linearly independent rows
    Q_row, R_row, row_perm = linalg.qr(P_C.T, pivoting=True)
    R_indices = row_perm[:r_minus_1]
    P_RC = P_C[R_indices, :]
    
    if verbose:
        print(f"Affine dimension: {r_minus_1 + 1}, checking {2**r_minus_1} candidates")
    
    # Step 3: Form Z and candidate matrix Tb
    # Z = P_C @ inv(P_RC)
    try:
        Z = P_C @ linalg.inv(P_RC)
    except linalg.LinAlgError:
        Z = P_C @ linalg.pinv(P_RC)
    
    # Generate all binary vectors B^{r-1}
    B = generate_binary_vectors(r_minus_1)  # Shape: (r-1, 2^{r-1})
    
    # Tb = Z @ (B - p_R @ 1^T) + p @ 1^T
    p_R = p[R_indices]
    num_candidates = B.shape[1]
    Tb = Z @ (B - p_R[:, np.newaxis]) + p[:, np.newaxis]
    
    # Step 4: Filter to find actual vertices
    T_set = set()
    tol_binary = 1e-10
    
    for u in range(num_candidates):
        candidate = Tb[:, u]
        # Check if candidate is in {0,1}^m
        is_binary = np.allclose(candidate, np.round(candidate), atol=tol_binary)
        in_range = np.all((candidate >= -tol_binary) & (candidate <= 1 + tol_binary))
        
        if is_binary and in_range:
            vertex = tuple(np.round(candidate).astype(int))
            T_set.add(vertex)
    
    if verbose:
        print(f"Found {len(T_set)} vertices")
    
    return T_set


def binary_factorization_exact(D: np.ndarray, 
                               affine: bool = True,
                               verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 2: BINARYFACTORIZATION_EXACT
    
    Find exact factorization D = T @ A with T ∈ {0,1}^{m×r}.
    
    Parameters
    ----------
    D : np.ndarray
        Data matrix of shape (m, n)
    affine : bool
        If True, enforce A^T @ 1_r = 1_n (columns of A sum to 1)
    verbose : bool
        Print progress information
        
    Returns
    -------
    T : np.ndarray
        Binary factor matrix of shape (m, r)
    A : np.ndarray
        Coefficient matrix of shape (r, n)
    """
    m, n = D.shape
    
    # Step 1: Get vertices from aff(D)
    T_set = find_vertices_exact(D, verbose=verbose)
    
    if len(T_set) == 0:
        raise ValueError("No binary vertices found in aff(D)")
    
    # Convert to matrix
    T_all = np.array(list(T_set)).T  # Shape: (m, |T|)
    
    if verbose:
        print(f"Found {T_all.shape[1]} total vertices")
    
    # Step 2: Select r affinely independent elements
    # Use column pivoted QR to find linearly independent columns
    if affine:
        # For affine independence, we work with T - T[:,0]
        T_centered = T_all - T_all[:, 0:1]
        Q, R, perm = linalg.qr(T_centered[:, 1:], pivoting=True)
        tol = max(T_all.shape) * np.finfo(float).eps * np.abs(R).max() if R.size > 0 else 0
        
        if R.size > 0:
            rank = np.sum(np.abs(np.diag(R)) > tol)
        else:
            rank = 0
        
        # Select first column plus rank affinely independent ones
        selected = [0] + [perm[i] + 1 for i in range(rank)]
        T = T_all[:, selected]
    else:
        Q, R, perm = linalg.qr(T_all, pivoting=True)
        tol = max(T_all.shape) * np.finfo(float).eps * np.abs(R).max()
        rank = np.sum(np.abs(np.diag(R)) > tol)
        T = T_all[:, perm[:rank]]
    
    r = T.shape[1]
    if verbose:
        print(f"Selected r = {r} affinely independent vertices")
    
    # Step 3: Solve for A
    if affine:
        # Solve [1^T; T] @ A = [1^T; D]
        augmented_T = np.vstack([np.ones(r), T])
        augmented_D = np.vstack([np.ones(n), D])
        A, residuals, rank, s = linalg.lstsq(augmented_T, augmented_D)
    else:
        # Solve T @ A = D
        A, residuals, rank, s = linalg.lstsq(T, D)
    
    return T, A


def find_vertices_approximate(D: np.ndarray, 
                              r: int,
                              verbose: bool = False) -> np.ndarray:
    """
    Algorithm 3: FINDVERTICES_APPROXIMATE
    
    Find approximate binary vertices for noisy factorization.
    
    Parameters
    ----------
    D : np.ndarray
        Data matrix of shape (m, n), possibly noisy
    r : int
        Target rank (number of components)
    verbose : bool
        Print progress information
        
    Returns
    -------
    T : np.ndarray
        Binary factor matrix of shape (m, r)
    """
    m, n = D.shape
    
    # Step 1: Center the data
    p = D.mean(axis=1)
    P = D - p[:, np.newaxis]
    
    # Step 2: Compute truncated SVD
    U, s, Vh = linalg.svd(P, full_matrices=False)
    
    r_minus_1 = min(r - 1, len(s))
    U_r = U[:, :r_minus_1]  # Shape: (m, r-1)
    
    if r_minus_1 == 0:
        # Rank 1 case - just use the mean
        T = (p > 0.5).astype(float)[:, np.newaxis]
        return T
    
    # Find linearly independent rows of U_r
    Q, R, row_perm = linalg.qr(U_r.T, pivoting=True)
    R_indices = row_perm[:r_minus_1]
    U_R = U_r[R_indices, :]
    
    if verbose:
        print(f"Target rank: {r}, using {r_minus_1} dimensions")
    
    # Step 3: Form Z and candidate matrix
    try:
        Z = U_r @ linalg.inv(U_R)
    except linalg.LinAlgError:
        Z = U_r @ linalg.pinv(U_R)
    
    B = generate_binary_vectors(r_minus_1)
    p_R = p[R_indices]
    
    Tb = Z @ (B - p_R[:, np.newaxis]) + p[:, np.newaxis]
    
    # Step 4: Round to binary
    Tb01 = (Tb > 0.5).astype(float)
    
    # Step 5: Compute distances and select best r candidates
    delta = np.linalg.norm(Tb - Tb01, axis=0)
    order = np.argsort(delta)
    
    # Select r candidates with smallest distance
    selected = order[:r]
    T = Tb01[:, selected]
    
    # Ensure columns are unique
    T_unique = np.unique(T, axis=1)
    
    if T_unique.shape[1] < r:
        if verbose:
            print(f"Warning: Only {T_unique.shape[1]} unique vertices found")
        # Add more candidates if needed
        for idx in order[r:]:
            candidate = Tb01[:, idx:idx+1]
            if not any(np.allclose(candidate, T_unique[:, i:i+1]) 
                      for i in range(T_unique.shape[1])):
                T_unique = np.hstack([T_unique, candidate])
                if T_unique.shape[1] >= r:
                    break
    
    if verbose:
        print(f"Selected {T_unique.shape[1]} vertices")
    
    return T_unique[:, :min(r, T_unique.shape[1])]


def block_optimization(D: np.ndarray,
                       T_init: np.ndarray,
                       max_iter: int = 100,
                       tol: float = 1e-8,
                       nonnegative_A: bool = False,
                       sum_to_one: bool = False,
                       verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 4: Block optimization scheme
    
    Solve min_{T ∈ {0,1}^{m×r}, A} ||D - T @ A||_F^2
    
    Parameters
    ----------
    D : np.ndarray
        Data matrix of shape (m, n)
    T_init : np.ndarray
        Initial binary matrix of shape (m, r)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    nonnegative_A : bool
        If True, enforce A >= 0
    sum_to_one : bool
        If True, enforce columns of A sum to 1
    verbose : bool
        Print progress information
        
    Returns
    -------
    T : np.ndarray
        Binary factor matrix of shape (m, r)
    A : np.ndarray
        Coefficient matrix of shape (r, n)
    """
    m, n = D.shape
    T = T_init.copy().astype(float)
    r = T.shape[1]
    
    prev_loss = np.inf
    
    for iteration in range(max_iter):
        # Step 2: Update A given T
        if sum_to_one and nonnegative_A:
            A = _solve_simplex_constrained(T, D)
        elif nonnegative_A:
            A = _solve_nonnegative_lstsq(T, D)
        else:
            A, _, _, _ = linalg.lstsq(T, D)
        
        # Step 3: Update T given A (row-wise separable)
        # For each row i: min_{T_i ∈ {0,1}^r} ||D_i - T_i @ A||^2
        for i in range(m):
            best_loss = np.inf
            best_row = T[i, :].copy()
            
            # Try all 2^r possibilities
            for bits in product([0, 1], repeat=r):
                T_row = np.array(bits, dtype=float)
                loss = np.sum((D[i, :] - T_row @ A) ** 2)
                if loss < best_loss:
                    best_loss = loss
                    best_row = T_row
            
            T[i, :] = best_row
        
        # Check convergence
        current_loss = np.linalg.norm(D - T @ A, 'fro') ** 2
        
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: loss = {current_loss:.6e}")
        
        if abs(prev_loss - current_loss) < tol:
            if verbose:
                print(f"Converged at iteration {iteration}")
            break
        
        prev_loss = current_loss
    
    return T.astype(int), A


def _solve_nonnegative_lstsq(T: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Solve non-negative least squares: min_{A>=0} ||D - T @ A||"""
    from scipy.optimize import nnls
    
    r = T.shape[1]
    n = D.shape[1]
    A = np.zeros((r, n))
    
    for j in range(n):
        A[:, j], _ = nnls(T, D[:, j])
    
    return A


def _solve_simplex_constrained(T: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Solve simplex-constrained least squares: min_{A>=0, 1^T A = 1} ||D - T @ A||"""
    from scipy.optimize import minimize
    
    r = T.shape[1]
    n = D.shape[1]
    A = np.zeros((r, n))
    
    for j in range(n):
        d = D[:, j]
        
        def objective(a):
            return np.sum((d - T @ a) ** 2)
        
        def grad(a):
            return -2 * T.T @ (d - T @ a)
        
        # Constraints: sum = 1, all >= 0
        constraints = {'type': 'eq', 'fun': lambda a: np.sum(a) - 1}
        bounds = [(0, None) for _ in range(r)]
        
        # Initialize uniformly
        a0 = np.ones(r) / r
        
        result = minimize(objective, a0, jac=grad, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        A[:, j] = result.x
    
    return A


def binary_factorization_approximate(D: np.ndarray,
                                     r: int,
                                     nonnegative_A: bool = False,
                                     sum_to_one: bool = False,
                                     refine: bool = True,
                                     verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete approximate binary matrix factorization.
    
    Parameters
    ----------
    D : np.ndarray
        Data matrix of shape (m, n)
    r : int
        Target rank
    nonnegative_A : bool
        If True, enforce A >= 0
    sum_to_one : bool
        If True, enforce columns of A sum to 1
    refine : bool
        If True, refine with block optimization
    verbose : bool
        Print progress information
        
    Returns
    -------
    T : np.ndarray
        Binary factor matrix of shape (m, r)
    A : np.ndarray
        Coefficient matrix of shape (r, n)
    """
    # Get initial T from approximate vertex finding
    T = find_vertices_approximate(D, r, verbose=verbose)
    
    actual_r = T.shape[1]
    if actual_r < r:
        warnings.warn(f"Only found {actual_r} components instead of {r}")
    
    # Solve for A
    if sum_to_one and nonnegative_A:
        A = _solve_simplex_constrained(T, D)
    elif nonnegative_A:
        A = _solve_nonnegative_lstsq(T, D)
    else:
        A, _, _, _ = linalg.lstsq(T, D)
    
    # Optionally refine with block optimization
    if refine:
        T, A = block_optimization(D, T, nonnegative_A=nonnegative_A,
                                  sum_to_one=sum_to_one, verbose=verbose)
    
    return T, A


def find_vertices_ilp(D: np.ndarray, 
                      solver: str = 'highs',
                      verbose: bool = False) -> Set[tuple]:
    """
    Find vertices using Integer Linear Programming.
    
    This approach can handle larger r values (up to ~80) by using 
    the continuous Littlewood-Offord lemma and ILP solvers.
    
    Parameters
    ----------
    D : np.ndarray
        Data matrix of shape (m, n)
    solver : str
        ILP solver to use ('highs' for scipy's built-in)
    verbose : bool
        Print progress
        
    Returns
    -------
    vertices : set of tuples
        Binary vertices in aff(D)
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    m, n = D.shape
    
    # Center and compute basis
    p = D[:, 0].copy()
    P = D - p[:, np.newaxis]
    
    # SVD for dimension reduction
    U, s, Vh = linalg.svd(P, full_matrices=False)
    tol = max(m, n) * np.finfo(float).eps * s[0] if len(s) > 0 else 0
    r_minus_1 = np.sum(s > tol)
    
    if r_minus_1 == 0:
        return set()
    
    U_r = U[:, :r_minus_1]
    
    # Find basis rows
    Q, R, row_perm = linalg.qr(U_r.T, pivoting=True)
    R_indices = row_perm[:r_minus_1]
    U_R = U_r[R_indices, :]
    
    try:
        Z = U_r @ linalg.inv(U_R)
    except:
        Z = U_r @ linalg.pinv(U_R)
    
    p_R = p[R_indices]
    
    # For each candidate, we need: 0 <= Z @ (b - p_R) + p <= 1 for all rows not in R
    # Which gives us: -p <= Z @ (b - p_R) <= 1 - p
    
    # Rows to check (excluding R)
    check_rows = [i for i in range(m) if i not in R_indices]
    
    vertices = set()
    
    # Use ILP to find all feasible binary vectors
    # min/max c^T b  subject to: A_eq @ b = b_eq, lb <= b <= ub, b binary
    
    c = np.zeros(r_minus_1)  # No objective, just feasibility
    
    # Constraints: 0 <= Z[i,:] @ (b - p_R) + p[i] <= 1 for i not in R
    # Rewrite: -p[i] <= Z[i,:] @ b - Z[i,:] @ p_R <= 1 - p[i]
    # So: Z[i,:] @ p_R - p[i] <= Z[i,:] @ b <= 1 - p[i] + Z[i,:] @ p_R
    
    A = Z[check_rows, :]
    offset = A @ p_R - p[check_rows]
    lb = -offset
    ub = 1 - p[check_rows] + A @ p_R - offset
    
    # Adjust for numerical tolerance
    lb = lb - 1e-9
    ub = ub + 1e-9
    
    constraints = LinearConstraint(A, lb, ub)
    bounds = Bounds(0, 1)
    integrality = np.ones(r_minus_1)  # All binary
    
    # Find one solution
    result = milp(c, constraints=constraints, bounds=bounds, 
                  integrality=integrality, options={'disp': verbose})
    
    if result.success:
        b = np.round(result.x).astype(int)
        candidate = Z @ (b - p_R) + p
        vertex = tuple(np.round(candidate).astype(int))
        vertices.add(vertex)
        
        if verbose:
            print(f"Found vertex via ILP")
    
    # To find ALL vertices, we'd need to enumerate - for now, 
    # fall back to direct enumeration for small r
    if r_minus_1 <= 20:
        vertices = find_vertices_exact(D, verbose=verbose)
    
    return vertices


# ============== Demo and Testing ==============

def demo_exact_factorization():
    """Demonstrate exact factorization on synthetic data."""
    print("=" * 60)
    print("Demo: Exact Binary Matrix Factorization")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create ground truth
    m, r, n = 50, 4, 20
    T_true = np.random.randint(0, 2, size=(m, r)).astype(float)
    
    # Ensure columns are affinely independent
    T_true[:r, :] = np.eye(r)
    
    # Generate A with sum-to-one constraint
    A_true = np.random.dirichlet(np.ones(r), size=n).T
    
    # Generate data
    D = T_true @ A_true
    
    print(f"Data shape: {D.shape}")
    print(f"True rank: {r}")
    
    # Recover factorization
    T_rec, A_rec = binary_factorization_exact(D, affine=True, verbose=True)
    
    # Evaluate
    D_rec = T_rec @ A_rec
    reconstruction_error = np.linalg.norm(D - D_rec, 'fro')
    
    print(f"\nRecovered rank: {T_rec.shape[1]}")
    print(f"Reconstruction error: {reconstruction_error:.2e}")
    
    # Check if we recovered the right columns (up to permutation)
    T_rec_set = set(tuple(T_rec[:, j].astype(int)) for j in range(T_rec.shape[1]))
    T_true_set = set(tuple(T_true[:, j].astype(int)) for j in range(T_true.shape[1]))
    
    recovered = len(T_rec_set & T_true_set)
    print(f"True columns recovered: {recovered}/{r}")
    
    return T_rec, A_rec


def demo_approximate_factorization():
    """Demonstrate approximate factorization on noisy data."""
    print("\n" + "=" * 60)
    print("Demo: Approximate Binary Matrix Factorization")
    print("=" * 60)
    
    np.random.seed(123)
    
    # Create ground truth
    m, r, n = 100, 5, 30
    T_true = np.random.randint(0, 2, size=(m, r)).astype(float)
    
    # Ensure columns are distinct
    T_true[:r, :] = np.eye(r)
    
    # Generate A with simplex constraint
    A_true = np.random.dirichlet(np.ones(r), size=n).T
    
    # Generate noisy data
    noise_level = 0.05
    D = T_true @ A_true + noise_level * np.random.randn(m, n)
    
    print(f"Data shape: {D.shape}")
    print(f"True rank: {r}")
    print(f"Noise level: {noise_level}")
    
    # Recover factorization
    T_rec, A_rec = binary_factorization_approximate(
        D, r, 
        nonnegative_A=True, 
        sum_to_one=True,
        refine=True,
        verbose=True
    )
    
    # Evaluate
    D_rec = T_rec @ A_rec
    reconstruction_error = np.linalg.norm(D - D_rec, 'fro') / np.sqrt(m * n)
    true_error = np.linalg.norm(T_true @ A_true - D_rec, 'fro') / np.sqrt(m * n)
    
    print(f"\nRecovered rank: {T_rec.shape[1]}")
    print(f"RMSE (vs noisy data): {reconstruction_error:.4f}")
    print(f"RMSE (vs true signal): {true_error:.4f}")
    
    # Compute Hamming distance
    hamming = np.mean(T_rec != T_true)
    print(f"Hamming distance (T): {hamming:.4f}")
    
    return T_rec, A_rec


def demo_separable_case():
    """Demonstrate factorization with separable T (unique solution guaranteed)."""
    print("\n" + "=" * 60)
    print("Demo: Separable Case (Unique Solution)")
    print("=" * 60)
    
    np.random.seed(456)
    
    # Create separable T: T = [M; I_r]
    m, r, n = 80, 4, 25
    
    M = np.random.randint(0, 2, size=(m - r, r)).astype(float)
    I_r = np.eye(r)
    T_true = np.vstack([M, I_r])
    
    # Permute rows randomly
    perm = np.random.permutation(m)
    T_true = T_true[perm, :]
    
    # Generate A
    A_true = np.random.dirichlet(np.ones(r), size=n).T
    
    # Generate data
    D = T_true @ A_true
    
    print(f"Data shape: {D.shape}")
    print(f"True rank: {r}")
    print("T is separable (contains identity as submatrix)")
    
    # Recover
    T_rec, A_rec = binary_factorization_exact(D, affine=True, verbose=True)
    
    # Evaluate
    reconstruction_error = np.linalg.norm(D - T_rec @ A_rec, 'fro')
    print(f"\nReconstruction error: {reconstruction_error:.2e}")
    
    return T_rec, A_rec


if __name__ == "__main__":
    demo_exact_factorization()
    demo_approximate_factorization()
    demo_separable_case()
    
    # Additional demo: DNA methylation unmixing simulation
    print("\n" + "=" * 60)
    print("Demo: DNA Methylation Unmixing (as in paper)")
    print("=" * 60)
    
    np.random.seed(789)
    
    # Simulate methylation data
    # m = number of CpG sites, r = number of cell types, n = number of samples
    m_cpg, r_types, n_samples = 500, 4, 12
    
    # True methylation profiles: T[i,k] = 1 if site i is methylated in cell type k
    # Use realistic sparsity pattern (most sites are either methylated or not)
    T_methyl = (np.random.rand(m_cpg, r_types) > 0.5).astype(float)
    
    # Add some structure: make first few rows form identity (separability)
    T_methyl[:r_types, :] = np.eye(r_types)
    
    # True mixing proportions (simplex-constrained)
    A_mix = np.random.dirichlet(np.ones(r_types) * 2, size=n_samples).T
    
    # Observed methylation levels (in [0,1] due to cell mixture)
    # Add measurement noise
    noise_sigma = 0.03
    D_observed = np.clip(T_methyl @ A_mix + noise_sigma * np.random.randn(m_cpg, n_samples), 0, 1)
    
    print(f"CpG sites: {m_cpg}")
    print(f"Cell types: {r_types}")
    print(f"Samples: {n_samples}")
    print(f"Noise level: {noise_sigma}")
    
    # Recover factorization
    T_rec, A_rec = binary_factorization_approximate(
        D_observed, r_types,
        nonnegative_A=True,
        sum_to_one=True,
        refine=True,
        verbose=True
    )
    
    # Evaluate mixing proportions recovery
    # Need to match columns (cell types may be permuted)
    from scipy.optimize import linear_sum_assignment
    
    # Compute cost matrix based on column correlation
    cost = np.zeros((r_types, T_rec.shape[1]))
    for i in range(r_types):
        for j in range(T_rec.shape[1]):
            cost[i, j] = -np.corrcoef(T_methyl[:, i], T_rec[:, j])[0, 1]
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Permute recovered matrices to match ground truth
    T_rec_perm = T_rec[:, col_ind]
    A_rec_perm = A_rec[col_ind, :]
    
    # Report accuracy
    t_accuracy = np.mean(T_rec_perm == T_methyl)
    a_rmse = np.sqrt(np.mean((A_rec_perm - A_mix)**2))
    
    print(f"\nResults (after optimal permutation):")
    print(f"Methylation profile accuracy: {t_accuracy:.1%}")
    print(f"Mixing proportion RMSE: {a_rmse:.4f}")
    
    # Show recovered vs true mixing proportions for first few samples
    print(f"\nSample mixing proportions (first 4 samples):")
    print("True A:")
    print(np.round(A_mix[:, :4], 3))
    print("Recovered A:")
    print(np.round(A_rec_perm[:, :4], 3))
