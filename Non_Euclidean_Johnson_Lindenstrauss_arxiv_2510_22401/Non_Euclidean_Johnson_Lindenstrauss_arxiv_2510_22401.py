"""
Implementation of: Johnson-Lindenstrauss Lemma Beyond Euclidean Geometry

ArXiv: https://arxiv.org/abs/2510.22401

Authors:
    - Chengyuan Deng
    - Jie Gao
    - Kevin Lu
    - Feng Luo
    - Cheng Xin

Abstract:
    The Johnson-Lindenstrauss (JL) lemma is a cornerstone of dimensionality reduction in Euclidean space, 
    but its applicability to non-Euclidean data has remained limited. This paper extends the JL lemma 
    beyond Euclidean geometry to handle general dissimilarity matrices that are prevalent in real-world 
    applications. We present two complementary approaches: First, we show the JL transform can be applied 
    to vectors in pseudo-Euclidean space with signature (p,q), providing theoretical guarantees that depend 
    on the ratio of the (p,q) norm and Euclidean norm of two vectors. Second, we prove that any symmetric 
    hollow dissimilarity matrix can be represented as a matrix of generalized power distances.

This implementation provides both the pseudo-Euclidean and power distance approaches for 
non-Euclidean Johnson-Lindenstrauss transforms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from sklearn.datasets import make_blobs
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

class NonEuclideanJL:
    """
    Non-Euclidean Johnson-Lindenstrauss Transform implementation.
    
    This class implements two methods for extending JL transforms to non-Euclidean data:
    1. Pseudo-Euclidean JL Transform
    2. Power Distance JL Transform
    """
    
    def __init__(self, epsilon: float = 0.5, random_state: Optional[int] = None):
        """
        Initialize the Non-Euclidean JL transformer.
        
        Args:
            epsilon: Distortion parameter (0 < epsilon < 1)
            random_state: Random seed for reproducibility
        """
        self.epsilon = epsilon
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_target_dimension(self, n: int) -> int:
        """
        Compute target dimension for JL transform: O(log n / epsilon^2)
        
        Args:
            n: Number of points
            
        Returns:
            Target dimension
        """
        return max(1, int(2 * np.log(n) / (self.epsilon ** 2)))
    
    def _compute_gram_matrix(self, D: np.ndarray) -> np.ndarray:
        """
        Compute Gram matrix from dissimilarity matrix.
        
        Args:
            D: Symmetric hollow dissimilarity matrix
            
        Returns:
            Gram matrix
        """
        n = D.shape[0]
        J = np.ones((n, n)) / n
        C = np.eye(n) - J  # Centering matrix
        return -0.5 * C @ D @ C
    
    def pseudo_euclidean_transform(self, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Apply pseudo-Euclidean JL transform.
        
        Args:
            D: Symmetric hollow dissimilarity matrix of shape (n, n)
            
        Returns:
            Tuple of (embedded_points, signature, info_dict)
        """
        n = D.shape[0]
        
        # Step 1: Compute Gram matrix and its eigendecomposition
        G = self._compute_gram_matrix(D)
        eigenvalues, eigenvectors = eigh(G)
        
        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 2: Determine (p,q) signature
        p = np.sum(eigenvalues >= 1e-10)  # Number of positive eigenvalues
        q = np.sum(eigenvalues < -1e-10)  # Number of negative eigenvalues
        
        # Step 3: Compute coordinates in pseudo-Euclidean space
        sqrt_abs_eigenvals = np.sqrt(np.abs(eigenvalues))
        X = (eigenvectors * sqrt_abs_eigenvals).T
        
        # Step 4: Apply JL transform to positive and negative parts separately
        target_dim = self._compute_target_dimension(n)
        
        if p > 0:
            # JL transform for positive part
            R_p = np.random.randn(target_dim, p) / np.sqrt(target_dim)
            X_p_proj = R_p @ X[:p, :]
        else:
            X_p_proj = np.zeros((target_dim, n))
        
        if q > 0:
            # JL transform for negative part  
            R_q = np.random.randn(target_dim, q) / np.sqrt(target_dim)
            X_q_proj = R_q @ X[p:p+q, :]
        else:
            X_q_proj = np.zeros((target_dim, n))
        
        # Combine projections
        embedded_points = np.vstack([X_p_proj, X_q_proj]).T
        signature = (target_dim, target_dim)  # (p', q')
        
        info = {
            'original_signature': (p, q),
            'target_dimension': 2 * target_dim,
            'eigenvalues': eigenvalues,
            'gram_matrix': G
        }
        
        return embedded_points, signature, info
    
    def power_distance_transform(self, D: np.ndarray) -> Tuple[np.ndarray, float, dict]:
        """
        Apply power distance JL transform.
        
        Args:
            D: Symmetric hollow dissimilarity matrix of shape (n, n)
            
        Returns:
            Tuple of (embedded_points, radius, info_dict)
        """
        n = D.shape[0]
        
        # Step 1: Compute Gram matrix and find smallest eigenvalue
        G = self._compute_gram_matrix(D)
        eigenvalues = eigh(G, eigvals_only=True)
        e_n = np.min(eigenvalues)
        
        # Step 2: Compute radius parameter
        r = np.sqrt(np.abs(e_n) / 2) if e_n < 0 else 0
        
        # Step 3: Create Euclidean distance matrix E
        I = np.eye(n)
        J = np.ones((n, n))
        E = D + 4 * r**2 * (I - J)
        
        # Step 4: Recover Euclidean coordinates from E
        G_E = self._compute_gram_matrix(E)
        eigenvals_E, eigenvecs_E = eigh(G_E)
        
        # Take only positive eigenvalues for Euclidean embedding
        pos_idx = eigenvals_E > 1e-10
        pos_eigenvals = eigenvals_E[pos_idx]
        pos_eigenvecs = eigenvecs_E[:, pos_idx]
        
        # Compute coordinates
        sqrt_eigenvals = np.sqrt(pos_eigenvals)
        X_euclidean = (pos_eigenvecs * sqrt_eigenvals).T
        
        # Step 5: Apply standard JL transform to Euclidean coordinates
        target_dim = self._compute_target_dimension(n)
        d_original = X_euclidean.shape[0]
        
        if d_original > 0:
            R = np.random.randn(target_dim, d_original) / np.sqrt(target_dim)
            embedded_points = (R @ X_euclidean).T
        else:
            embedded_points = np.zeros((n, target_dim))
        
        info = {
            'radius': r,
            'smallest_eigenvalue': e_n,
            'target_dimension': target_dim,
            'original_dimension': d_original,
            'euclidean_matrix': E
        }
        
        return embedded_points, r, info
    
    def compute_pairwise_distances(self, X: np.ndarray, signature: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Compute pairwise distances in the embedded space.
        
        Args:
            X: Embedded points
            signature: (p, q) signature for pseudo-Euclidean distance, None for Euclidean
            
        Returns:
            Pairwise distance matrix
        """
        n = X.shape[0]
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                diff = X[i] - X[j]
                
                if signature is not None:
                    # Pseudo-Euclidean distance
                    p_dim, q_dim = signature
                    if len(diff) >= p_dim + q_dim:
                        pos_part = diff[:p_dim]
                        neg_part = diff[p_dim:p_dim+q_dim]
                        dist_sq = np.sum(pos_part**2) - np.sum(neg_part**2)
                        dist = np.sqrt(np.abs(dist_sq)) if dist_sq >= 0 else 0
                    else:
                        dist = np.linalg.norm(diff)
                else:
                    # Euclidean distance
                    dist = np.linalg.norm(diff)
                
                distances[i, j] = distances[j, i] = dist
        
        return distances

def create_synthetic_dissimilarity_matrix(n: int, method: str = 'simplex') -> np.ndarray:
    """
    Create synthetic non-Euclidean dissimilarity matrices for testing.
    
    Args:
        n: Number of points
        method: 'simplex' or 'balls'
        
    Returns:
        Symmetric hollow dissimilarity matrix
    """
    if method == 'simplex':
        # Create simplex-based non-Euclidean data
        # Generate points where first n-1 coordinates form simplex, last coordinate dominates
        X = np.random.randn(n, n)
        
        # Make last coordinate dominate to create large negative eigenvalue
        X[:, -1] = 10 * np.random.randn(n)
        
        # Compute pairwise distances with modified metric
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                # Use a non-Euclidean distance that violates triangle inequality
                dist = np.sum((X[i] - X[j])**2) + 0.5 * (X[i, -1] - X[j, -1])**4
                D[i, j] = D[j, i] = dist
        
    elif method == 'balls':
        # Create ball-based dissimilarity (inspired by Delft's balls)
        centers = np.random.randn(n, 2) * 5
        radii = np.random.uniform(0.5, 2.0, n)
        
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                center_dist = np.linalg.norm(centers[i] - centers[j])
                # Distance between ball surfaces (can be negative)
                surface_dist = max(0, center_dist - radii[i] - radii[j])
                D[i, j] = D[j, i] = surface_dist**2
    
    return D

def visualize_eigenvalues(D: np.ndarray, title: str):
    """
    Visualize eigenvalue spectrum of Gram matrix.
    """
    jl = NonEuclideanJL()
    G = jl._compute_gram_matrix(D)
    eigenvals = eigh(G, eigvals_only=True)
    eigenvals = np.sort(eigenvals)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(eigenvals, 'bo-', markersize=4)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title(f'{title} - Eigenvalue Spectrum')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    pos_count = np.sum(eigenvals > 1e-10)
    neg_count = np.sum(eigenvals < -1e-10)
    zero_count = len(eigenvals) - pos_count - neg_count
    
    labels = ['Positive', 'Negative', 'Zero']
    counts = [pos_count, neg_count, zero_count]
    colors = ['green', 'red', 'gray']
    
    plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title(f'{title} - Eigenvalue Distribution\n(p={pos_count}, q={neg_count})')
    
    plt.tight_layout()
    plt.show()

def compare_distance_preservation(D_orig: np.ndarray, D_embedded: np.ndarray, method_name: str):
    """
    Compare original vs embedded distances.
    """
    # Extract upper triangular part (excluding diagonal)
    n = D_orig.shape[0]
    indices = np.triu_indices(n, k=1)
    orig_distances = D_orig[indices]
    embedded_distances = D_embedded[indices]
    
    plt.figure(figsize=(12, 4))
    
    # Scatter plot of original vs embedded distances
    plt.subplot(1, 3, 1)
    plt.scatter(orig_distances, embedded_distances, alpha=0.6, s=20)
    plt.plot([orig_distances.min(), orig_distances.max()], 
             [orig_distances.min(), orig_distances.max()], 'r--', alpha=0.8)
    plt.xlabel('Original Distances')
    plt.ylabel('Embedded Distances')
    plt.title(f'{method_name}\nOriginal vs Embedded')
    plt.grid(True, alpha=0.3)
    
    # Relative error distribution
    plt.subplot(1, 3, 2)
    rel_errors = np.abs(orig_distances - embedded_distances) / (orig_distances + 1e-10)
    plt.hist(rel_errors, bins=30, alpha=0.7, color='orange')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title(f'Relative Error Distribution\nMean: {np.mean(rel_errors):.3f}')
    plt.grid(True, alpha=0.3)
    
    # Error vs distance magnitude
    plt.subplot(1, 3, 3)
    plt.scatter(orig_distances, rel_errors, alpha=0.6, s=20, c='purple')
    plt.xlabel('Original Distance')
    plt.ylabel('Relative Error')
    plt.title('Error vs Distance Magnitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    max_rel_error = np.max(rel_errors)
    mean_rel_error = np.mean(rel_errors)
    print(f"{method_name} Statistics:")
    print(f"  Maximum relative error: {max_rel_error:.4f}")
    print(f"  Mean relative error: {mean_rel_error:.4f}")
    print(f"  Standard deviation: {np.std(rel_errors):.4f}")
    print()

def demonstrate_non_euclidean_jl():
    """
    Demonstrate both non-Euclidean JL methods on synthetic data.
    """
    print("=" * 60)
    print("Non-Euclidean Johnson-Lindenstrauss Transform Demo")
    print("=" * 60)
    
    # Create synthetic non-Euclidean data
    n = 100
    
    # Test on simplex data (good for pseudo-Euclidean method)
    print(f"\nCreating simplex-based dissimilarity matrix with {n} points...")
    D_simplex = create_synthetic_dissimilarity_matrix(n, method='simplex')
    
    # Test on ball data (good for power distance method)
    print(f"Creating ball-based dissimilarity matrix with {n} points...")
    D_balls = create_synthetic_dissimilarity_matrix(n, method='balls')
    
    # Visualize eigenvalue spectra
    visualize_eigenvalues(D_simplex, "Simplex Data")
    visualize_eigenvalues(D_balls, "Ball Data")
    
    # Initialize JL transformer
    jl = NonEuclideanJL(epsilon=0.3, random_state=42)
    
    print("\n" + "="*50)
    print("PSEUDO-EUCLIDEAN JL TRANSFORM")
    print("="*50)
    
    # Apply pseudo-Euclidean transform on simplex data
    print("\nApplying pseudo-Euclidean JL transform on simplex data...")
    X_pe_simplex, sig_pe, info_pe = jl.pseudo_euclidean_transform(D_simplex)
    
    print(f"Original signature (p, q): {info_pe['original_signature']}")
    print(f"Target dimension: {info_pe['target_dimension']}")
    print(f"Embedded points shape: {X_pe_simplex.shape}")
    
    # Compute distances in embedded space
    D_pe_simplex = jl.compute_pairwise_distances(X_pe_simplex, sig_pe)
    
    # Compare distance preservation
    compare_distance_preservation(D_simplex, D_pe_simplex, "Pseudo-Euclidean (Simplex)")
    
    print("\n" + "="*50)
    print("POWER DISTANCE JL TRANSFORM")
    print("="*50)
    
    # Apply power distance transform on ball data
    print("\nApplying power distance JL transform on ball data...")
    X_pd_balls, r_pd, info_pd = jl.power_distance_transform(D_balls)
    
    print(f"Radius parameter r: {r_pd:.4f}")
    print(f"Smallest eigenvalue: {info_pd['smallest_eigenvalue']:.4f}")
    print(f"Target dimension: {info_pd['target_dimension']}")
    print(f"Embedded points shape: {X_pd_balls.shape}")
    
    # Compute distances in embedded space
    D_pd_balls = jl.compute_pairwise_distances(X_pd_balls)
    
    # For power distance, we need to add back the radius term
    # since the embedded distances are between centers
    D_pd_balls_corrected = D_pd_balls**2 - 4 * r_pd**2
    D_pd_balls_corrected = np.maximum(D_pd_balls_corrected, 0)  # Ensure non-negative
    
    # Compare distance preservation
    compare_distance_preservation(D_balls, D_pd_balls_corrected, "Power Distance (Balls)")
    
    print("\n" + "="*50)
    print("CROSS-METHOD COMPARISON")
    print("="*50)
    
    # Compare both methods on the same data
    print("\nComparing both methods on simplex data...")
    
    # Pseudo-Euclidean on simplex (should work well)
    X_pe_simplex, _, _ = jl.pseudo_euclidean_transform(D_simplex)
    D_pe_simplex = jl.compute_pairwise_distances(X_pe_simplex, sig_pe)
    
    # Power distance on simplex (might have larger error)
    X_pd_simplex, r_pd_simplex, _ = jl.power_distance_transform(D_simplex)
    D_pd_simplex = jl.compute_pairwise_distances(X_pd_simplex)
    D_pd_simplex_corrected = D_pd_simplex**2 - 4 * r_pd_simplex**2
    D_pd_simplex_corrected = np.maximum(D_pd_simplex_corrected, 0)
    
    # Create comparison visualization
    plt.figure(figsize=(15, 5))
    
    # Extract upper triangular distances for comparison
    indices = np.triu_indices(n, k=1)
    orig_dist = D_simplex[indices]
    pe_dist = D_pe_simplex[indices]
    pd_dist = D_pd_simplex_corrected[indices]
    
    # Scatter plots
    plt.subplot(1, 3, 1)
    plt.scatter(orig_dist, pe_dist, alpha=0.6, color='blue', label='Pseudo-Euclidean')
    plt.plot([orig_dist.min(), orig_dist.max()], 
             [orig_dist.min(), orig_dist.max()], 'r--', alpha=0.8)
    plt.xlabel('Original Distances')
    plt.ylabel('Embedded Distances')
    plt.title('Pseudo-Euclidean Method')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(orig_dist, pd_dist, alpha=0.6, color='green', label='Power Distance')
    plt.plot([orig_dist.min(), orig_dist.max()], 
             [orig_dist.min(), orig_dist.max()], 'r--', alpha=0.8)
    plt.xlabel('Original Distances')
    plt.ylabel('Embedded Distances')
    plt.title('Power Distance Method')
    plt.grid(True, alpha=0.3)
    
    # Error comparison
    plt.subplot(1, 3, 3)
    pe_errors = np.abs(orig_dist - pe_dist) / (orig_dist + 1e-10)
    pd_errors = np.abs(orig_dist - pd_dist) / (orig_dist + 1e-10)
    
    plt.hist(pe_errors, bins=30, alpha=0.7, color='blue', label=f'PE (mean: {np.mean(pe_errors):.3f})')
    plt.hist(pd_errors, bins=30, alpha=0.7, color='green', label=f'PD (mean: {np.mean(pd_errors):.3f})')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Error Comparison on Simplex Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal Comparison (Simplex Data):")
    print(f"Pseudo-Euclidean mean relative error: {np.mean(pe_errors):.4f}")
    print(f"Power Distance mean relative error: {np.mean(pd_errors):.4f}")
    
    if np.mean(pe_errors) < np.mean(pd_errors):
        print("✓ Pseudo-Euclidean method performs better on this non-Euclidean data")
    else:
        print("✓ Power Distance method performs better on this non-Euclidean data")

if __name__ == "__main__":
    demonstrate_non_euclidean_jl()