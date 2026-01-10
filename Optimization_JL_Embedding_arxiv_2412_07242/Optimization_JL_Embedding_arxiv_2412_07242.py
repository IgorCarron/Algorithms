"""
Implementation of: Optimization Can Learn Johnson Lindenstrauss Embeddings

ArXiv: https://arxiv.org/abs/2412.07242

Authors:
    - Nikos Tsikouras
    - Constantine Caramanis
    - Christos Tzamos

Abstract:
    Embeddings play a pivotal role across various disciplines, offering compact representations of complex data structures. Randomized methods like Johnson-Lindenstrauss (JL) provide state-of-the-art and essentially unimprovable theoretical guarantees for achieving such representations. These guarantees are worst-case and in particular, neither the analysis, nor the algorithm, takes into account any potential structural information of the data. The natural question is: must we randomize? Could we instead use an optimization-based approach, working directly with the data? A first answer is no: as we show, the distance-preserving objective of JL has a non-convex landscape over the space of projection matrices, with many bad stationary points. But this is not the final answer. We present a novel method motivated by diffusion models, that circumvents this fundamental challenge: rather than performing optimization directly over the space of projection matrices, we use optimization over the larger space of random solution samplers, gradually reducing the variance of the sampler.

This implementation provides the optimization-based approach for learning JL embeddings by optimizing over Gaussian samplers and gradually reducing variance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class OptimizedJLEmbedding:
    """
    Implementation of optimization-based Johnson-Lindenstrauss embedding.
    
    This class implements the main algorithm from the paper which optimizes over
    the space of Gaussian solution samplers (M, σ²) rather than directly over
    projection matrices.
    """
    
    def __init__(self, n_data: int, d_input: int, k_output: int, epsilon: float = None):
        """
        Initialize the optimization-based JL embedding.
        
        Args:
            n_data: Number of data points
            d_input: Input dimension
            k_output: Output dimension (target dimension)
            epsilon: Distortion parameter (if None, set to O(sqrt(log n / k)))
        """
        self.n = n_data
        self.d = d_input
        self.k = k_output
        self.epsilon = epsilon if epsilon is not None else np.sqrt(np.log(n_data) / k_output)
        
        # Initialize parameters: mean matrix M and variance σ²
        self.M = np.zeros((k_output, d_input))  # Mean matrix
        self.sigma_sq = 1.0  # Variance parameter
        
        # Algorithm parameters
        self.rho = 1e-4  # Stationarity threshold
        self.learning_rate = 0.01
        
    def objective_function(self, X: np.ndarray, M: np.ndarray, sigma_sq: float) -> float:
        """
        Compute the objective function g(M, σ²) from Equation 4.
        
        This is a relaxed version that sums probabilities of constraint violations
        plus a regularization term.
        
        Args:
            X: Data matrix (n x d)
            M: Mean matrix (k x d)
            sigma_sq: Variance parameter
            
        Returns:
            Objective function value
        """
        n, d = X.shape
        k = M.shape[0]
        
        # Compute the probability of constraint violation for each data point
        total_prob = 0.0
        
        for i in range(n):
            x = X[i, :]
            
            # Expected squared norm after projection
            mean_norm_sq = np.sum((M @ x) ** 2)
            
            # Non-centrality parameter for chi-squared distribution
            delta = mean_norm_sq / sigma_sq if sigma_sq > 0 else mean_norm_sq / 1e-10
            
            # Lower and upper bounds for JL guarantee
            lower_bound = k * (1 - self.epsilon) / sigma_sq if sigma_sq > 0 else k * (1 - self.epsilon) / 1e-10
            upper_bound = k * (1 + self.epsilon) / sigma_sq if sigma_sq > 0 else k * (1 + self.epsilon) / 1e-10
            
            # Probability of violating JL constraint using chi-squared CDF
            # We approximate the non-central chi-squared with central chi-squared
            if delta < 1e-6:  # Nearly zero mean case
                prob_violation = chi2.cdf(lower_bound, k) + (1 - chi2.cdf(upper_bound, k))
            else:
                # Approximate non-central chi-squared
                # Use shifted central chi-squared approximation
                effective_df = k + 2 * delta
                prob_violation = chi2.cdf(lower_bound, effective_df) + (1 - chi2.cdf(upper_bound, effective_df))
            
            total_prob += min(prob_violation, 1.0)  # Clip probability
        
        # Add regularization term
        regularization = sigma_sq / 2.0
        
        return total_prob + regularization
    
    def compute_gradients(self, X: np.ndarray, M: np.ndarray, sigma_sq: float) -> Tuple[np.ndarray, float]:
        """
        Compute gradients of the objective function with respect to M and σ².
        
        Args:
            X: Data matrix (n x d)
            M: Mean matrix (k x d)
            sigma_sq: Variance parameter
            
        Returns:
            Tuple of (gradient w.r.t. M, gradient w.r.t. σ²)
        """
        n, d = X.shape
        k = M.shape[0]
        
        grad_M = np.zeros_like(M)
        grad_sigma = 0.0
        
        # Numerical gradient computation for simplicity
        eps_M = 1e-6
        eps_sigma = 1e-6
        
        # Gradient w.r.t. M
        for i in range(k):
            for j in range(d):
                M_plus = M.copy()
                M_plus[i, j] += eps_M
                
                M_minus = M.copy()
                M_minus[i, j] -= eps_M
                
                grad_M[i, j] = (self.objective_function(X, M_plus, sigma_sq) - 
                               self.objective_function(X, M_minus, sigma_sq)) / (2 * eps_M)
        
        # Gradient w.r.t. σ²
        obj_plus = self.objective_function(X, M, sigma_sq + eps_sigma)
        obj_minus = self.objective_function(X, M, sigma_sq - eps_sigma)
        grad_sigma = (obj_plus - obj_minus) / (2 * eps_sigma)
        
        return grad_M, grad_sigma
    
    def fit(self, X: np.ndarray, max_iterations: int = 1000, verbose: bool = False) -> List[dict]:
        """
        Optimize the JL embedding using the algorithm from the paper.
        
        Args:
            X: Data matrix (n x d) with unit norm rows
            max_iterations: Maximum number of optimization iterations
            verbose: Whether to print progress
            
        Returns:
            List of dictionaries containing optimization history
        """
        # Normalize data to unit norm
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        # Initialize from the known good sampler (0, 1)
        self.M = np.zeros((self.k, self.d))
        self.sigma_sq = 1.0
        
        history = []
        
        for iteration in range(max_iterations):
            # Compute objective and gradients
            obj_value = self.objective_function(X_normalized, self.M, self.sigma_sq)
            grad_M, grad_sigma = self.compute_gradients(X_normalized, self.M, self.sigma_sq)
            
            # Check for convergence (simplified)
            grad_norm = np.sqrt(np.sum(grad_M**2) + grad_sigma**2)
            
            # Store history
            history.append({
                'iteration': iteration,
                'objective': obj_value,
                'gradient_norm': grad_norm,
                'variance': self.sigma_sq,
                'matrix_norm': np.linalg.norm(self.M, 'fro')
            })
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: obj={obj_value:.6f}, grad_norm={grad_norm:.6f}, σ²={self.sigma_sq:.6f}")
            
            # Check convergence
            if grad_norm < self.rho and self.sigma_sq < 1e-3:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Gradient descent step
            self.M -= self.learning_rate * grad_M
            self.sigma_sq = max(1e-6, self.sigma_sq - self.learning_rate * grad_sigma)
        
        return history
    
    def compute_distortion(self, X: np.ndarray) -> float:
        """
        Compute the maximum distortion achieved by the learned embedding.
        
        Args:
            X: Data matrix (n x d)
            
        Returns:
            Maximum distortion
        """
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        # Project using the learned matrix M
        Y = self.M @ X_normalized.T
        
        max_distortion = 0.0
        for i in range(X_normalized.shape[0]):
            original_norm = np.linalg.norm(X_normalized[i])**2
            projected_norm = np.linalg.norm(Y[:, i])**2 / self.k
            
            distortion = abs(projected_norm - original_norm)
            max_distortion = max(max_distortion, distortion)
        
        return max_distortion
    
    def get_embedding_matrix(self) -> np.ndarray:
        """
        Get the final learned embedding matrix.
        
        Returns:
            The learned projection matrix M
        """
        return self.M.copy()

def generate_random_baseline_comparison(X: np.ndarray, k: int, n_trials: int = 100) -> Tuple[float, float]:
    """
    Generate baseline comparison using standard Gaussian random projection.
    
    Args:
        X: Data matrix
        k: Target dimension
        n_trials: Number of random trials
        
    Returns:
        Tuple of (average distortion, minimum distortion)
    """
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X_normalized.shape
    
    distortions = []
    
    for trial in range(n_trials):
        # Generate random Gaussian matrix
        A = np.random.normal(0, 1, (k, d))
        
        # Project data
        Y = A @ X_normalized.T
        
        # Compute maximum distortion
        max_distortion = 0.0
        for i in range(n):
            original_norm = np.linalg.norm(X_normalized[i])**2
            projected_norm = np.linalg.norm(Y[:, i])**2 / k
            
            distortion = abs(projected_norm - original_norm)
            max_distortion = max(max_distortion, distortion)
        
        distortions.append(max_distortion)
    
    return np.mean(distortions), np.min(distortions)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic dataset
    n_points = 100
    input_dim = 50
    target_dim = 10
    
    print(f"Generating synthetic dataset: {n_points} points in {input_dim}D -> {target_dim}D")
    
    # Generate random data points
    X = np.random.normal(0, 1, (n_points, input_dim))
    
    # Create and fit the optimized JL embedding
    print("\nFitting optimization-based JL embedding...")
    jl_optimizer = OptimizedJLEmbedding(n_points, input_dim, target_dim)
    history = jl_optimizer.fit(X, max_iterations=500, verbose=True)
    
    # Compute final distortion
    final_distortion = jl_optimizer.compute_distortion(X)
    print(f"\nFinal distortion (optimized): {final_distortion:.6f}")
    
    # Compare with random baseline
    print("\nComputing random baseline...")
    avg_random_distortion, min_random_distortion = generate_random_baseline_comparison(X, target_dim, n_trials=100)
    print(f"Average random distortion: {avg_random_distortion:.6f}")
    print(f"Minimum random distortion: {min_random_distortion:.6f}")
    
    # Extract data for plotting
    iterations = [h['iteration'] for h in history]
    objectives = [h['objective'] for h in history]
    variances = [h['variance'] for h in history]
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Objective function evolution
    plt.subplot(1, 3, 1)
    plt.plot(iterations, objectives, 'b-', linewidth=2, label='Optimized Objective')
    plt.axhline(y=avg_random_distortion, color='r', linestyle='--', linewidth=2, 
                label=f'Random Avg: {avg_random_distortion:.3f}')
    plt.axhline(y=min_random_distortion, color='g', linestyle='--', linewidth=2, 
                label=f'Random Min: {min_random_distortion:.3f}')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Objective Function Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Variance evolution
    plt.subplot(1, 3, 2)
    plt.plot(iterations, variances, 'purple', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Variance σ²')
    plt.title('Variance Reduction Over Time')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Distortion comparison
    plt.subplot(1, 3, 3)
    methods = ['Optimized\nMethod', 'Random\nAverage', 'Random\nMinimum']
    distortions = [final_distortion, avg_random_distortion, min_random_distortion]
    colors = ['blue', 'red', 'green']
    
    bars = plt.bar(methods, distortions, color=colors, alpha=0.7)
    plt.ylabel('Maximum Distortion')
    plt.title('Distortion Comparison')
    plt.yscale('log')
    
    # Add value labels on bars
    for bar, dist in zip(bars, distortions):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{dist:.4f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create a second figure showing the convergence behavior in detail
    plt.figure(figsize=(12, 4))
    
    # Plot convergence metrics
    plt.subplot(1, 2, 1)
    gradient_norms = [h['gradient_norm'] for h in history]
    plt.semilogy(iterations, gradient_norms, 'orange', linewidth=2)
    plt.axhline(y=jl_optimizer.rho, color='red', linestyle='--', alpha=0.7, 
                label=f'Convergence threshold: {jl_optimizer.rho}')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title('Gradient Norm Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot matrix norm evolution
    plt.subplot(1, 2, 2)
    matrix_norms = [h['matrix_norm'] for h in history]
    plt.plot(iterations, matrix_norms, 'teal', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Matrix Frobenius Norm')
    plt.title('Learned Matrix Evolution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final results summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Dataset: {n_points} points, {input_dim}D -> {target_dim}D")
    print(f"Target distortion (ε): {jl_optimizer.epsilon:.4f}")
    print(f"Final variance (σ²): {jl_optimizer.sigma_sq:.6f}")
    print(f"Iterations to convergence: {len(history)}")
    print("\nDistortion Results:")
    print(f"  Optimized method: {final_distortion:.6f}")
    print(f"  Random average:   {avg_random_distortion:.6f}")
    print(f"  Random minimum:   {min_random_distortion:.6f}")
    print(f"\nImprovement over random average: {(avg_random_distortion/final_distortion):.2f}x")
    print(f"Improvement over random minimum: {(min_random_distortion/final_distortion):.2f}x")
    
    # Verify that the learned matrix satisfies JL guarantee
    theoretical_bound = jl_optimizer.epsilon
    satisfies_jl = final_distortion <= theoretical_bound
    print(f"\nJL Guarantee Check:")
    print(f"  Theoretical bound: {theoretical_bound:.6f}")
    print(f"  Achieved distortion: {final_distortion:.6f}")
    print(f"  Satisfies JL guarantee: {'✓ YES' if satisfies_jl else '✗ NO'}")
