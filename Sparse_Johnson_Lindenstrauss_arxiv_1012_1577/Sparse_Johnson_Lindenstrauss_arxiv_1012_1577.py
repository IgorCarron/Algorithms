"""Implementation of: Sparser Johnson-Lindenstrauss Transforms

ArXiv: https://arxiv.org/abs/1012.1577

Authors:
    - Daniel M. Kane
    - Jelani Nelson

Abstract:
    We give two different and simple constructions for dimensionality reduction in $\ell_2$ via linear mappings that are sparse: only an $O(\varepsilon)$-fraction of entries in each column of our embedding matrices are non-zero to achieve distortion $1+\varepsilon$ with high probability, while still achieving the asymptotically optimal number of rows. These are the first constructions to provide subconstant sparsity for all values of parameters, improving upon previous works of Achlioptas (JCSS 2003) and Dasgupta, Kumar, and Sarlós (STOC 2010).

This implementation provides both block and graph constructions for sparse Johnson-Lindenstrauss transforms with optimal sparsity parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import hashlib
import warnings
warnings.filterwarnings('ignore')


class SparseJohnsonLindenstrauss:
    """Sparse Johnson-Lindenstrauss transform implementation.
    
    Implements both block and graph constructions from the paper achieving
    sparsity s = O(ε^-1 log(1/δ)) and target dimension k = O(ε^-2 log(1/δ)).
    """
    
    def __init__(self, 
                 d: int, 
                 epsilon: float = 0.1, 
                 delta: float = 0.1, 
                 construction: str = 'block',
                 random_seed: Optional[int] = None):
        """
        Initialize the sparse JL transform.
        
        Args:
            d: Input dimension
            epsilon: Distortion parameter (target: preserve norms within 1±ε)
            delta: Failure probability
            construction: 'block' or 'graph' construction
            random_seed: Random seed for reproducibility
        """
        self.d = d
        self.epsilon = epsilon
        self.delta = delta
        self.construction = construction
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Compute optimal parameters from the paper
        self.k = self._compute_target_dimension()
        self.s = self._compute_sparsity()
        
        # Create the sparse embedding matrix
        self.S = self._create_embedding_matrix()
        
    def _compute_target_dimension(self) -> int:
        """Compute target dimension k = Θ(ε^-2 log(1/δ))."""
        # Use the constant C = 8 as suggested in the paper
        k = max(1, int(8 * np.log(1/self.delta) / (self.epsilon**2)))
        return k
        
    def _compute_sparsity(self) -> int:
        """Compute sparsity s = Θ(ε^-1 log(1/δ))."""
        # Use constant to ensure s ≥ 2(2ε-ε²)^-1 log(1/δ) as in the paper
        min_s = max(1, int(2 * np.log(1/self.delta) / (2*self.epsilon - self.epsilon**2)))
        s = max(min_s, int(4 * np.log(1/self.delta) / self.epsilon))
        return min(s, self.k)  # s cannot exceed k
        
    def _create_embedding_matrix(self) -> np.ndarray:
        """Create the sparse embedding matrix based on construction type."""
        if self.construction == 'block':
            return self._create_block_construction()
        elif self.construction == 'graph':
            return self._create_graph_construction()
        else:
            raise ValueError("Construction must be 'block' or 'graph'")
            
    def _create_block_construction(self) -> np.ndarray:
        """Create embedding matrix using block construction (Figure 1c).
        
        Divide target vector into s contiguous blocks of size k/s each.
        Each input coordinate is hashed to one location in each block.
        """
        S = np.zeros((self.k, self.d))
        
        # Number of blocks
        num_blocks = self.s
        block_size = self.k // num_blocks
        
        # For each input coordinate
        for j in range(self.d):
            # Hash to one location in each block
            for block_idx in range(num_blocks):
                # Use simple hash function for demonstration
                hash_val = int(hashlib.md5(f"{j}_{block_idx}".encode()).hexdigest(), 16)
                location = block_idx * block_size + (hash_val % block_size)
                
                if location < self.k:
                    # Random sign ±1
                    sign = 1 if (hash_val >> 8) % 2 == 0 else -1
                    S[location, j] = sign / np.sqrt(self.s)
                    
        return S
        
    def _create_graph_construction(self) -> np.ndarray:
        """Create embedding matrix using graph construction (Figure 1b).
        
        Each input coordinate is hashed to exactly s target coordinates
        without replacement.
        """
        S = np.zeros((self.k, self.d))
        
        # For each input coordinate
        for j in range(self.d):
            # Hash to s distinct locations without replacement
            # Use deterministic sampling based on coordinate index
            np.random.seed(j)  # Deterministic per coordinate
            locations = np.random.choice(self.k, size=min(self.s, self.k), replace=False)
            
            for loc in locations:
                # Random sign ±1
                sign = 1 if np.random.rand() < 0.5 else -1
                S[loc, j] = sign / np.sqrt(self.s)
                
        # Reset random seed
        np.random.seed(None)
        return S
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the sparse JL transform to input data.
        
        Args:
            X: Input data of shape (d,) or (d, n) where n is number of vectors
            
        Returns:
            Transformed data of shape (k,) or (k, n)
        """
        if X.ndim == 1:
            return self.S @ X
        else:
            return self.S @ X
            
    def compute_distortion(self, x: np.ndarray) -> float:
        """Compute the distortion ||Sx||²/||x||² for a given vector."""
        if np.linalg.norm(x) == 0:
            return 0.0
        
        Sx = self.transform(x)
        return np.linalg.norm(Sx)**2 / np.linalg.norm(x)**2
        
    def get_sparsity_ratio(self) -> float:
        """Get the fraction of non-zero entries in the embedding matrix."""
        return np.count_nonzero(self.S) / (self.k * self.d)
        
    def get_column_sparsity(self) -> np.ndarray:
        """Get number of non-zero entries per column."""
        return np.count_nonzero(self.S, axis=0)


def test_jl_property(sjl: SparseJohnsonLindenstrauss, 
                     num_vectors: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Test the Johnson-Lindenstrauss property on random vectors.
    
    Returns:
        Tuple of (norms_original, norms_transformed)
    """
    norms_orig = []
    norms_transformed = []
    distortions = []
    
    for _ in range(num_vectors):
        # Generate random unit vector
        x = np.random.randn(sjl.d)
        x = x / np.linalg.norm(x)
        
        # Transform
        Sx = sjl.transform(x)
        
        # Record norms
        norms_orig.append(np.linalg.norm(x))
        norms_transformed.append(np.linalg.norm(Sx))
        distortions.append(sjl.compute_distortion(x))
        
    return np.array(norms_orig), np.array(norms_transformed), np.array(distortions)


def compare_constructions(d: int = 1000, 
                         epsilon: float = 0.1, 
                         delta: float = 0.1) -> None:
    """Compare block and graph constructions."""
    # Create both constructions
    sjl_block = SparseJohnsonLindenstrauss(d, epsilon, delta, 'block', random_seed=42)
    sjl_graph = SparseJohnsonLindenstrauss(d, epsilon, delta, 'graph', random_seed=42)
    
    print(f"Input dimension: {d}")
    print(f"Target dimension (k): {sjl_block.k}")
    print(f"Sparsity parameter (s): {sjl_block.s}")
    print(f"\nBlock construction sparsity ratio: {sjl_block.get_sparsity_ratio():.4f}")
    print(f"Graph construction sparsity ratio: {sjl_graph.get_sparsity_ratio():.4f}")
    
    # Test JL property
    _, _, dist_block = test_jl_property(sjl_block, 500)
    _, _, dist_graph = test_jl_property(sjl_graph, 500)
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Sparsity patterns
    plt.subplot(1, 3, 1)
    plt.spy(sjl_block.S[:100, :100], markersize=0.5)
    plt.title(f'Block Construction Sparsity Pattern\n(s={sjl_block.s}, k={sjl_block.k})')
    plt.xlabel('Input Coordinates')
    plt.ylabel('Output Coordinates')
    
    # Plot 2: Distortion comparison
    plt.subplot(1, 3, 2)
    plt.hist(dist_block, alpha=0.6, bins=30, label=f'Block (violations: {np.sum(np.abs(dist_block - 1) > epsilon)/len(dist_block)*100:.1f}%)')
    plt.hist(dist_graph, alpha=0.6, bins=30, label=f'Graph (violations: {np.sum(np.abs(dist_graph - 1) > epsilon)/len(dist_graph)*100:.1f}%)')
    plt.axvline(1-epsilon, color='red', linestyle='--', alpha=0.7, label=f'Target bounds: 1±{epsilon}')
    plt.axvline(1+epsilon, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Distortion ||Sx||²/||x||²')
    plt.ylabel('Frequency')
    plt.title('Distortion Distribution Comparison')
    plt.legend()
    
    # Plot 3: Column sparsity
    plt.subplot(1, 3, 3)
    col_sparse_block = sjl_block.get_column_sparsity()
    col_sparse_graph = sjl_graph.get_column_sparsity()
    plt.hist(col_sparse_block, alpha=0.6, bins=20, label=f'Block (mean: {np.mean(col_sparse_block):.1f})')
    plt.hist(col_sparse_graph, alpha=0.6, bins=20, label=f'Graph (mean: {np.mean(col_sparse_graph):.1f})')
    plt.xlabel('Non-zeros per Column')
    plt.ylabel('Frequency')
    plt.title('Column Sparsity Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def study_parameter_scaling():
    """Study how sparsity scales with epsilon and delta parameters."""
    d = 500
    epsilons = [0.05, 0.1, 0.2, 0.3, 0.5]
    deltas = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    # Study epsilon scaling
    sparsities_eps = []
    dimensions_eps = []
    for eps in epsilons:
        sjl = SparseJohnsonLindenstrauss(d, eps, 0.1, 'block')
        sparsities_eps.append(sjl.s)
        dimensions_eps.append(sjl.k)
    
    # Study delta scaling  
    sparsities_delta = []
    dimensions_delta = []
    for delta in deltas:
        sjl = SparseJohnsonLindenstrauss(d, 0.1, delta, 'block')
        sparsities_delta.append(sjl.s)
        dimensions_delta.append(sjl.k)
    
    plt.figure(figsize=(12, 4))
    
    # Plot epsilon scaling
    plt.subplot(1, 2, 1)
    plt.loglog(epsilons, sparsities_eps, 'bo-', label='Sparsity s')
    plt.loglog(epsilons, dimensions_eps, 'ro-', label='Dimension k')
    plt.loglog(epsilons, 1/np.array(epsilons), 'b--', alpha=0.5, label='O(1/ε)')
    plt.loglog(epsilons, 1/np.array(epsilons)**2, 'r--', alpha=0.5, label='O(1/ε²)')
    plt.xlabel('Distortion Parameter ε')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Scaling vs Epsilon\n(δ = 0.1 fixed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot delta scaling
    plt.subplot(1, 2, 2)
    log_deltas = -np.log(deltas)
    plt.plot(log_deltas, sparsities_delta, 'bo-', label='Sparsity s')
    plt.plot(log_deltas, dimensions_delta, 'ro-', label='Dimension k')
    plt.plot(log_deltas, log_deltas * 10, 'b--', alpha=0.5, label='O(log(1/δ))')
    plt.xlabel('log(1/δ)')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Scaling vs Delta\n(ε = 0.1 fixed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_application():
    """Demonstrate application to approximate matrix multiplication."""
    # Create random matrices
    d, n, m = 200, 100, 80
    A = np.random.randn(d, n)
    B = np.random.randn(d, m)
    
    # Exact product
    exact_product = A.T @ B
    
    # Approximate using sparse JL
    epsilon = 0.2
    sjl = SparseJohnsonLindenstrauss(d, epsilon, 0.1, 'block', random_seed=42)
    
    SA = sjl.transform(A)
    SB = sjl.transform(B)
    approx_product = SA.T @ SB
    
    # Compute error
    error_frobenius = np.linalg.norm(exact_product - approx_product, 'fro')
    relative_error = error_frobenius / np.linalg.norm(exact_product, 'fro')
    
    print(f"\nMatrix Multiplication Approximation:")
    print(f"Original matrices: A ({d}x{n}), B ({d}x{m})")
    print(f"Sketched matrices: SA ({sjl.k}x{n}), SB ({sjl.k}x{m})")
    print(f"Dimension reduction ratio: {sjl.k/d:.3f}")
    print(f"Sparsity ratio: {sjl.get_sparsity_ratio():.4f}")
    print(f"Relative error: {relative_error:.4f}")
    print(f"Target error bound: ~{3*epsilon:.3f} (from Theorem 21)")
    
    # Visualize error distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(exact_product - approx_product), cmap='Reds')
    plt.colorbar(label='Absolute Error')
    plt.title(f'Error Matrix |A^T B - SA^T SB|\nRelative Error: {relative_error:.3f}')
    
    plt.subplot(1, 2, 2)
    errors_flat = np.abs(exact_product - approx_product).flatten()
    plt.hist(errors_flat, bins=30, alpha=0.7)
    plt.axvline(np.mean(errors_flat), color='red', linestyle='--', label=f'Mean: {np.mean(errors_flat):.3f}')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pointwise Errors')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Sparse Johnson-Lindenstrauss Transforms Implementation")
    print("=" * 55)
    
    # Demonstrate the constructions
    print("\n1. Comparing Block and Graph Constructions:")
    compare_constructions(d=1000, epsilon=0.1, delta=0.1)
    
    # Study parameter scaling
    print("\n2. Parameter Scaling Analysis:")
    study_parameter_scaling()
    
    # Application demonstration
    print("\n3. Application to Matrix Multiplication:")
    demonstrate_application()
    
    print("\nImplementation complete! This demonstrates the key results from")
    print("Kane & Nelson (2012): sparse JL transforms with s = O(ε^-1 log(1/δ))")
    print("sparsity and k = O(ε^-2 log(1/δ)) target dimension.")