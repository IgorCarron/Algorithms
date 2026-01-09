"""
Compositional Kernel Search for Gaussian Process Regression

Implementation of:
"Structure Discovery in Nonparametric Regression through Compositional Kernel Search"
by Duvenaud, Lloyd, Grosse, Tenenbaum, Ghahramani (ICML 2013)

Paper: https://arxiv.org/abs/1302.4922

This implementation provides:
1. Base kernels: SE, Periodic, Linear, RQ
2. Compositional kernel algebra (+ and ×)
3. Greedy search over kernel structures using BIC
4. Posterior decomposition for interpretability
5. Visualization utilities
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular, cho_solve
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import copy
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Base Kernel Classes
# =============================================================================

class Kernel(ABC):
    """Abstract base class for all kernels."""
    
    def __init__(self, dim: int = 0):
        self.dim = dim  # Input dimension this kernel operates on
        self._params = {}
    
    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix K(X1, X2)."""
        pass
    
    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Return parameters as flat array (log-transformed for optimization)."""
        pass
    
    @abstractmethod
    def set_params(self, params: np.ndarray):
        """Set parameters from flat array (log-transformed)."""
        pass
    
    @abstractmethod
    def num_params(self) -> int:
        """Return number of parameters."""
        pass
    
    @abstractmethod
    def copy(self) -> 'Kernel':
        """Return a deep copy of the kernel."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Return string representation."""
        pass
    
    def _extract_dim(self, X: np.ndarray) -> np.ndarray:
        """Extract the relevant dimension from input."""
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X[:, self.dim:self.dim+1]


class SquaredExponential(Kernel):
    """
    Squared Exponential (RBF) kernel.
    k(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
    
    Encodes smooth functions with local correlations.
    """
    
    def __init__(self, dim: int = 0, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__(dim)
        self.lengthscale = lengthscale
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        x1 = self._extract_dim(X1)
        x2 = self._extract_dim(X2)
        sq_dist = (x1 - x2.T) ** 2
        return self.variance * np.exp(-sq_dist / (2 * self.lengthscale ** 2))
    
    def get_params(self) -> np.ndarray:
        return np.array([np.log(self.lengthscale), np.log(self.variance)])
    
    def set_params(self, params: np.ndarray):
        self.lengthscale = np.exp(params[0])
        self.variance = np.exp(params[1])
    
    def num_params(self) -> int:
        return 2
    
    def copy(self) -> 'SquaredExponential':
        return SquaredExponential(self.dim, self.lengthscale, self.variance)
    
    def __str__(self) -> str:
        return f"SE_{self.dim}"


class Periodic(Kernel):
    """
    Periodic kernel.
    k(x, x') = σ² exp(-2 sin²(π|x - x'|/p) / ℓ²)
    
    Encodes repeating patterns with period p.
    """
    
    def __init__(self, dim: int = 0, lengthscale: float = 1.0, 
                 period: float = 1.0, variance: float = 1.0):
        super().__init__(dim)
        self.lengthscale = lengthscale
        self.period = period
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        x1 = self._extract_dim(X1)
        x2 = self._extract_dim(X2)
        diff = x1 - x2.T
        sin_arg = np.pi * np.abs(diff) / self.period
        return self.variance * np.exp(-2 * np.sin(sin_arg) ** 2 / self.lengthscale ** 2)
    
    def get_params(self) -> np.ndarray:
        return np.array([np.log(self.lengthscale), np.log(self.period), np.log(self.variance)])
    
    def set_params(self, params: np.ndarray):
        self.lengthscale = np.exp(params[0])
        self.period = np.exp(params[1])
        self.variance = np.exp(params[2])
    
    def num_params(self) -> int:
        return 3
    
    def copy(self) -> 'Periodic':
        return Periodic(self.dim, self.lengthscale, self.period, self.variance)
    
    def __str__(self) -> str:
        return f"Per_{self.dim}"


class Linear(Kernel):
    """
    Linear kernel.
    k(x, x') = σ²_b + σ²_v (x - ℓ)(x' - ℓ)
    
    Encodes linear functions (Bayesian linear regression).
    """
    
    def __init__(self, dim: int = 0, variance_bias: float = 1.0,
                 variance_slope: float = 1.0, offset: float = 0.0):
        super().__init__(dim)
        self.variance_bias = variance_bias
        self.variance_slope = variance_slope
        self.offset = offset
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        x1 = self._extract_dim(X1)
        x2 = self._extract_dim(X2)
        return self.variance_bias + self.variance_slope * (x1 - self.offset) * (x2 - self.offset).T
    
    def get_params(self) -> np.ndarray:
        return np.array([np.log(self.variance_bias + 1e-6), 
                        np.log(self.variance_slope), 
                        self.offset])
    
    def set_params(self, params: np.ndarray):
        self.variance_bias = np.exp(params[0])
        self.variance_slope = np.exp(params[1])
        self.offset = params[2]
    
    def num_params(self) -> int:
        return 3
    
    def copy(self) -> 'Linear':
        return Linear(self.dim, self.variance_bias, self.variance_slope, self.offset)
    
    def __str__(self) -> str:
        return f"Lin_{self.dim}"


class RationalQuadratic(Kernel):
    """
    Rational Quadratic kernel.
    k(x, x') = σ² (1 + ||x - x'||² / (2αℓ²))^(-α)
    
    Equivalent to infinite mixture of SE kernels with different lengthscales.
    Good for multi-scale variation.
    """
    
    def __init__(self, dim: int = 0, lengthscale: float = 1.0, 
                 alpha: float = 1.0, variance: float = 1.0):
        super().__init__(dim)
        self.lengthscale = lengthscale
        self.alpha = alpha
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        x1 = self._extract_dim(X1)
        x2 = self._extract_dim(X2)
        sq_dist = (x1 - x2.T) ** 2
        return self.variance * (1 + sq_dist / (2 * self.alpha * self.lengthscale ** 2)) ** (-self.alpha)
    
    def get_params(self) -> np.ndarray:
        return np.array([np.log(self.lengthscale), np.log(self.alpha), np.log(self.variance)])
    
    def set_params(self, params: np.ndarray):
        self.lengthscale = np.exp(params[0])
        self.alpha = np.exp(params[1])
        self.variance = np.exp(params[2])
    
    def num_params(self) -> int:
        return 3
    
    def copy(self) -> 'RationalQuadratic':
        return RationalQuadratic(self.dim, self.lengthscale, self.alpha, self.variance)
    
    def __str__(self) -> str:
        return f"RQ_{self.dim}"


class WhiteNoise(Kernel):
    """
    White noise kernel (for observation noise).
    k(x, x') = σ² δ(x, x')
    """
    
    def __init__(self, variance: float = 0.1):
        super().__init__(0)
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if X1.shape[0] == X2.shape[0] and np.allclose(X1, X2):
            return self.variance * np.eye(X1.shape[0])
        return np.zeros((X1.shape[0], X2.shape[0]))
    
    def get_params(self) -> np.ndarray:
        return np.array([np.log(self.variance)])
    
    def set_params(self, params: np.ndarray):
        self.variance = np.exp(params[0])
    
    def num_params(self) -> int:
        return 1
    
    def copy(self) -> 'WhiteNoise':
        return WhiteNoise(self.variance)
    
    def __str__(self) -> str:
        return "WN"


# =============================================================================
# Composite Kernel Classes (Sum and Product)
# =============================================================================

class SumKernel(Kernel):
    """Sum of two kernels: k(x,x') = k1(x,x') + k2(x,x')"""
    
    def __init__(self, k1: Kernel, k2: Kernel):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.k1(X1, X2) + self.k2(X1, X2)
    
    def get_params(self) -> np.ndarray:
        return np.concatenate([self.k1.get_params(), self.k2.get_params()])
    
    def set_params(self, params: np.ndarray):
        n1 = self.k1.num_params()
        self.k1.set_params(params[:n1])
        self.k2.set_params(params[n1:])
    
    def num_params(self) -> int:
        return self.k1.num_params() + self.k2.num_params()
    
    def copy(self) -> 'SumKernel':
        return SumKernel(self.k1.copy(), self.k2.copy())
    
    def __str__(self) -> str:
        return f"({self.k1} + {self.k2})"
    
    def get_components(self) -> List[Kernel]:
        """Get list of additive components (for decomposition)."""
        components = []
        if isinstance(self.k1, SumKernel):
            components.extend(self.k1.get_components())
        else:
            components.append(self.k1)
        if isinstance(self.k2, SumKernel):
            components.extend(self.k2.get_components())
        else:
            components.append(self.k2)
        return components


class ProductKernel(Kernel):
    """Product of two kernels: k(x,x') = k1(x,x') × k2(x,x')"""
    
    def __init__(self, k1: Kernel, k2: Kernel):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.k1(X1, X2) * self.k2(X1, X2)
    
    def get_params(self) -> np.ndarray:
        return np.concatenate([self.k1.get_params(), self.k2.get_params()])
    
    def set_params(self, params: np.ndarray):
        n1 = self.k1.num_params()
        self.k1.set_params(params[:n1])
        self.k2.set_params(params[n1:])
    
    def num_params(self) -> int:
        return self.k1.num_params() + self.k2.num_params()
    
    def copy(self) -> 'ProductKernel':
        return ProductKernel(self.k1.copy(), self.k2.copy())
    
    def __str__(self) -> str:
        s1 = str(self.k1)
        s2 = str(self.k2)
        # Add parentheses around sums for clarity
        if isinstance(self.k1, SumKernel):
            s1 = f"({s1})"
        if isinstance(self.k2, SumKernel):
            s2 = f"({s2})"
        return f"{s1} × {s2}"


# =============================================================================
# Gaussian Process Class
# =============================================================================

class GaussianProcess:
    """
    Gaussian Process for regression with analytical marginal likelihood.
    """
    
    def __init__(self, kernel: Kernel, noise_variance: float = 0.1):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.L = None
        self.alpha = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the GP to training data."""
        self.X_train = X.reshape(-1, 1) if X.ndim == 1 else X
        self.y_train = y.reshape(-1, 1) if y.ndim == 1 else y
        
        # Compute kernel matrix with noise
        K = self.kernel(self.X_train, self.X_train)
        K += self.noise_variance * np.eye(len(X))
        K += 1e-6 * np.eye(len(X))  # Jitter for numerical stability
        
        # Cholesky decomposition
        try:
            self.L = cholesky(K, lower=True)
            self.alpha = cho_solve((self.L, True), self.y_train)
        except np.linalg.LinAlgError:
            # Fallback to direct inverse
            self.K_inv = np.linalg.inv(K)
            self.alpha = self.K_inv @ self.y_train
            self.L = None
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance at test points.
        
        Returns:
            mean: Posterior mean
            var: Posterior variance
        """
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        K_star = self.kernel(X, self.X_train)
        K_star_star = self.kernel(X, X)
        
        # Posterior mean
        mean = K_star @ self.alpha
        
        # Posterior variance
        if self.L is not None:
            v = solve_triangular(self.L, K_star.T, lower=True)
            var = np.diag(K_star_star) - np.sum(v ** 2, axis=0)
        else:
            var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
        
        var = np.maximum(var, 1e-10)  # Ensure positive variance
        
        return mean.flatten(), var
    
    def log_marginal_likelihood(self) -> float:
        """
        Compute log marginal likelihood: log p(y|X, θ)
        
        = -0.5 * y^T K^{-1} y - 0.5 * log|K| - n/2 * log(2π)
        """
        n = len(self.y_train)
        
        if self.L is not None:
            # Using Cholesky
            log_det = 2 * np.sum(np.log(np.diag(self.L)))
            data_fit = -0.5 * self.y_train.T @ self.alpha
        else:
            # Direct computation
            K = self.kernel(self.X_train, self.X_train)
            K += self.noise_variance * np.eye(n) + 1e-6 * np.eye(n)
            sign, log_det = np.linalg.slogdet(K)
            if sign <= 0:
                return -np.inf
            data_fit = -0.5 * self.y_train.T @ self.K_inv @ self.y_train
        
        complexity = -0.5 * log_det
        constant = -0.5 * n * np.log(2 * np.pi)
        
        return float(data_fit + complexity + constant)
    
    def bic(self) -> float:
        """
        Compute Bayesian Information Criterion.
        BIC = -2 * log_likelihood + k * log(n)
        
        Lower is better.
        """
        n = len(self.y_train)
        k = self.kernel.num_params() + 1  # +1 for noise variance
        log_lik = self.log_marginal_likelihood()
        return -2 * log_lik + k * np.log(n)
    
    def optimize(self, n_restarts: int = 3, max_iter: int = 100) -> float:
        """
        Optimize kernel hyperparameters by maximizing log marginal likelihood.
        
        Returns:
            Best log marginal likelihood achieved.
        """
        def neg_log_lik(params):
            # Set kernel params
            n_kernel = self.kernel.num_params()
            self.kernel.set_params(params[:n_kernel])
            self.noise_variance = np.exp(params[n_kernel])
            
            # Recompute
            try:
                self.fit(self.X_train, self.y_train)
                ll = self.log_marginal_likelihood()
                return -ll if np.isfinite(ll) else 1e10
            except:
                return 1e10
        
        best_params = None
        best_ll = -np.inf
        
        # Get current params
        current_kernel_params = self.kernel.get_params()
        current_noise = np.log(self.noise_variance)
        
        for restart in range(n_restarts):
            if restart == 0:
                # First restart: use current parameters
                x0 = np.concatenate([current_kernel_params, [current_noise]])
            else:
                # Random restarts
                x0 = np.concatenate([
                    current_kernel_params + 0.5 * np.random.randn(len(current_kernel_params)),
                    [current_noise + 0.5 * np.random.randn()]
                ])
            
            try:
                result = minimize(
                    neg_log_lik, x0,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'disp': False}
                )
                
                if -result.fun > best_ll:
                    best_ll = -result.fun
                    best_params = result.x
            except:
                continue
        
        if best_params is not None:
            neg_log_lik(best_params)
        
        return best_ll


# =============================================================================
# Kernel Search Algorithm
# =============================================================================

class KernelSearchResult:
    """Container for search results."""
    def __init__(self, kernel: Kernel, bic: float, log_lik: float, noise_var: float):
        self.kernel = kernel
        self.bic = bic
        self.log_lik = log_lik
        self.noise_var = noise_var
    
    def __str__(self):
        return f"Kernel: {self.kernel}, BIC: {self.bic:.2f}, LogLik: {self.log_lik:.2f}"


def get_base_kernels(dim: int = 0) -> List[Kernel]:
    """Get list of base kernel families for a given dimension."""
    return [
        SquaredExponential(dim),
        Periodic(dim),
        Linear(dim),
        RationalQuadratic(dim)
    ]


def expand_kernel(kernel: Kernel, base_kernels: List[Kernel]) -> List[Kernel]:
    """
    Generate all possible expansions of a kernel.
    
    Search operators:
    1. Replace S with S + B
    2. Replace S with S × B
    3. Replace base kernel B with B'
    """
    expansions = []
    
    # Add and multiply by base kernels
    for base in base_kernels:
        # S + B
        expansions.append(SumKernel(kernel.copy(), base.copy()))
        # S × B
        expansions.append(ProductKernel(kernel.copy(), base.copy()))
    
    return expansions


def evaluate_kernel(kernel: Kernel, X: np.ndarray, y: np.ndarray,
                   n_restarts: int = 3, max_iter: int = 100) -> KernelSearchResult:
    """Evaluate a kernel structure on data."""
    gp = GaussianProcess(kernel.copy())
    gp.fit(X, y)
    gp.optimize(n_restarts=n_restarts, max_iter=max_iter)
    
    bic = gp.bic()
    log_lik = gp.log_marginal_likelihood()
    
    return KernelSearchResult(gp.kernel, bic, log_lik, gp.noise_variance)


class CompositionalKernelSearch:
    """
    Greedy search over compositional kernel structures.
    
    Implements the algorithm from:
    "Structure Discovery in Nonparametric Regression through Compositional Kernel Search"
    Duvenaud et al., ICML 2013
    """
    
    def __init__(self, max_depth: int = 5, n_restarts: int = 3, 
                 verbose: bool = True, use_rq: bool = True):
        """
        Args:
            max_depth: Maximum search depth
            n_restarts: Number of random restarts for optimization
            verbose: Print progress
            use_rq: Include RQ kernel in search
        """
        self.max_depth = max_depth
        self.n_restarts = n_restarts
        self.verbose = verbose
        self.use_rq = use_rq
        self.search_history = []
    
    def get_base_kernels(self, n_dims: int) -> List[Kernel]:
        """Get base kernels for all dimensions."""
        kernels = []
        for d in range(n_dims):
            kernels.append(SquaredExponential(d))
            kernels.append(Periodic(d))
            kernels.append(Linear(d))
            if self.use_rq:
                kernels.append(RationalQuadratic(d))
        return kernels
    
    def search(self, X: np.ndarray, y: np.ndarray) -> KernelSearchResult:
        """
        Perform greedy search over kernel structures.
        
        Args:
            X: Input data (n_samples,) or (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            Best kernel structure found.
        """
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        n_dims = X.shape[1]
        
        base_kernels = self.get_base_kernels(n_dims)
        
        if self.verbose:
            print("=" * 60)
            print("Compositional Kernel Search")
            print("=" * 60)
            print(f"Data: {X.shape[0]} samples, {n_dims} dimensions")
            print(f"Max depth: {self.max_depth}")
            print(f"Base kernels: {[str(k) for k in base_kernels[:4]]}")
            print()
        
        # Initialize: evaluate all base kernels
        if self.verbose:
            print("Depth 0: Evaluating base kernels...")
        
        candidates = []
        for kernel in base_kernels:
            result = evaluate_kernel(kernel, X, y, self.n_restarts)
            candidates.append(result)
            if self.verbose:
                print(f"  {result}")
        
        # Select best
        best = min(candidates, key=lambda r: r.bic)
        self.search_history.append(('base', [str(c.kernel) for c in candidates], str(best.kernel), best.bic))
        
        if self.verbose:
            print(f"\nBest at depth 0: {best.kernel} (BIC: {best.bic:.2f})")
            print()
        
        # Greedy search
        for depth in range(1, self.max_depth + 1):
            if self.verbose:
                print(f"Depth {depth}: Expanding {best.kernel}...")
            
            # Generate expansions
            expansions = expand_kernel(best.kernel, base_kernels)
            
            # Evaluate expansions
            candidates = []
            for kernel in expansions:
                result = evaluate_kernel(kernel, X, y, self.n_restarts)
                candidates.append(result)
                if self.verbose:
                    print(f"  {result}")
            
            # Find best expansion
            best_expansion = min(candidates, key=lambda r: r.bic)
            
            self.search_history.append((
                f'depth_{depth}', 
                [str(c.kernel) for c in candidates[:5]],  # Top 5
                str(best_expansion.kernel), 
                best_expansion.bic
            ))
            
            # Check if improvement
            if best_expansion.bic < best.bic:
                best = best_expansion
                if self.verbose:
                    print(f"\nBest at depth {depth}: {best.kernel} (BIC: {best.bic:.2f})")
                    print()
            else:
                if self.verbose:
                    print(f"\nNo improvement at depth {depth}. Stopping.")
                break
        
        if self.verbose:
            print("=" * 60)
            print(f"Final kernel: {best.kernel}")
            print(f"BIC: {best.bic:.2f}")
            print(f"Log Marginal Likelihood: {best.log_lik:.2f}")
            print("=" * 60)
        
        return best


# =============================================================================
# Posterior Decomposition
# =============================================================================

def decompose_posterior(gp: GaussianProcess, X_test: np.ndarray) -> dict:
    """
    Decompose GP posterior into additive components.
    
    For a kernel k = k1 + k2, the posterior can be decomposed into
    independent components f1 and f2 where f = f1 + f2.
    
    Args:
        gp: Trained GaussianProcess
        X_test: Test points
        
    Returns:
        Dictionary with component names and (mean, var) tuples
    """
    if not isinstance(gp.kernel, SumKernel):
        # Single component
        mean, var = gp.predict(X_test)
        return {str(gp.kernel): (mean, var)}
    
    components = gp.kernel.get_components()
    results = {}
    
    X = gp.X_train
    y = gp.y_train.flatten()
    
    # Full kernel matrix
    K_full = gp.kernel(X, X) + gp.noise_variance * np.eye(len(X))
    K_full_inv = np.linalg.inv(K_full + 1e-6 * np.eye(len(X)))
    
    for comp in components:
        # Component kernel matrices
        K_comp = comp(X, X)
        K_comp_star = comp(X_test.reshape(-1, 1), X)
        K_comp_star_star = comp(X_test.reshape(-1, 1), X_test.reshape(-1, 1))
        
        # Conditional mean: μ_i + K_i^T (K_1 + K_2)^{-1} (f - μ_1 - μ_2)
        mean = K_comp_star @ K_full_inv @ y
        
        # Conditional variance: K_i - K_i^T (K_1 + K_2)^{-1} K_i
        var = np.diag(K_comp_star_star) - np.diag(K_comp_star @ K_full_inv @ K_comp_star.T)
        var = np.maximum(var, 1e-10)
        
        results[str(comp)] = (mean, np.sqrt(var))
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_gp_results(X: np.ndarray, y: np.ndarray, gp: GaussianProcess, 
                   X_test: Optional[np.ndarray] = None,
                   title: str = "GP Regression"):
    """
    Plot GP regression results with uncertainty.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    if X_test is None:
        x_min, x_max = X.min(), X.max()
        margin = 0.2 * (x_max - x_min)
        X_test = np.linspace(x_min - margin, x_max + margin, 200)
    
    mean, var = gp.predict(X_test)
    std = np.sqrt(var)
    
    plt.figure(figsize=(12, 6))
    
    # Plot data
    plt.scatter(X, y, c='black', s=20, zorder=5, label='Data')
    
    # Plot mean
    plt.plot(X_test, mean, 'b-', lw=2, label='Mean prediction')
    
    # Plot uncertainty
    plt.fill_between(X_test, mean - 2*std, mean + 2*std, 
                     alpha=0.2, color='blue', label='95% CI')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{title}\nKernel: {gp.kernel}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()


def plot_decomposition(X: np.ndarray, y: np.ndarray, gp: GaussianProcess,
                      X_test: Optional[np.ndarray] = None):
    """
    Plot posterior decomposition into additive components.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    if X_test is None:
        x_min, x_max = X.min(), X.max()
        X_test = np.linspace(x_min, x_max, 200)
    
    components = decompose_posterior(gp, X_test)
    n_components = len(components)
    
    fig, axes = plt.subplots(n_components + 2, 1, figsize=(12, 3*(n_components + 2)))
    
    # Full posterior
    mean, var = gp.predict(X_test)
    std = np.sqrt(var)
    
    axes[0].scatter(X, y, c='black', s=20, zorder=5)
    axes[0].plot(X_test, mean, 'b-', lw=2)
    axes[0].fill_between(X_test, mean - 2*std, mean + 2*std, alpha=0.2, color='blue')
    axes[0].set_title(f'Full Model: {gp.kernel}')
    axes[0].grid(True, alpha=0.3)
    
    # Individual components
    for i, (name, (comp_mean, comp_std)) in enumerate(components.items()):
        axes[i+1].plot(X_test, comp_mean, 'b-', lw=2)
        axes[i+1].fill_between(X_test, comp_mean - 2*comp_std, 
                               comp_mean + 2*comp_std, alpha=0.2, color='blue')
        axes[i+1].set_title(f'Component: {name}')
        axes[i+1].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y - gp.predict(X.reshape(-1, 1))[0]
    axes[-1].scatter(X, residuals, c='black', s=20)
    axes[-1].axhline(y=0, color='r', linestyle='--')
    axes[-1].set_title('Residuals')
    axes[-1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Example Datasets
# =============================================================================

def generate_airline_data():
    """
    Generate synthetic airline passenger data (similar to Box-Jenkins).
    Monthly totals with trend and seasonality.
    """
    np.random.seed(42)
    t = np.arange(0, 144)  # 12 years of monthly data
    
    # Components: trend + seasonality + noise
    trend = 100 + 2.5 * t
    seasonality = 30 * np.sin(2 * np.pi * t / 12)
    # Growing amplitude
    seasonality *= (1 + 0.01 * t)
    noise = 10 * np.random.randn(len(t))
    
    y = trend + seasonality + noise
    return t, y


def generate_mauna_loa_style():
    """
    Generate synthetic CO2 data (similar to Mauna Loa).
    Long-term trend + annual cycle + medium-term variations.
    """
    np.random.seed(123)
    t = np.arange(0, 500)
    
    # Long-term quadratic trend
    trend = 0.5 * t + 0.001 * t**2
    
    # Annual cycle
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    
    # Medium-term variations (RQ-like)
    medium = 3 * np.sin(2 * np.pi * t / 60) * np.exp(-0.002 * t)
    
    noise = 0.5 * np.random.randn(len(t))
    
    y = trend + seasonal + medium + noise
    return t, y


def generate_periodic_local():
    """Generate locally periodic data."""
    np.random.seed(456)
    t = np.linspace(0, 10, 200)
    
    # Periodic with local modulation
    y = np.sin(2 * np.pi * t) * np.exp(-0.1 * (t - 5)**2)
    y += 0.1 * np.random.randn(len(t))
    
    return t, y


# =============================================================================
# Main Demo
# =============================================================================

def demo():
    """Run demonstration of compositional kernel search."""
    print("\n" + "="*70)
    print(" COMPOSITIONAL KERNEL SEARCH DEMO")
    print(" Implementation of Duvenaud et al., ICML 2013")
    print("="*70 + "\n")
    
    # Generate data
    print("Generating synthetic airline-style data...")
    X, y = generate_airline_data()
    
    # Normalize for stability
    X_mean, X_std = X.mean(), X.std()
    y_mean, y_std = y.mean(), y.std()
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    # Run kernel search
    searcher = CompositionalKernelSearch(max_depth=4, n_restarts=2, verbose=True)
    result = searcher.search(X_norm, y_norm)
    
    # Create GP with best kernel
    gp = GaussianProcess(result.kernel, noise_variance=result.noise_var)
    gp.fit(X_norm, y_norm)
    
    # Predictions
    X_test = np.linspace(X_norm.min() - 0.5, X_norm.max() + 1.0, 300)
    mean, var = gp.predict(X_test)
    
    # Denormalize
    X_test_orig = X_test * X_std + X_mean
    mean_orig = mean * y_std + y_mean
    std_orig = np.sqrt(var) * y_std
    
    print("\n" + "-"*50)
    print("Prediction Summary:")
    print(f"  Training range: {X.min():.1f} to {X.max():.1f}")
    print(f"  Prediction range: {X_test_orig.min():.1f} to {X_test_orig.max():.1f}")
    print(f"  Final kernel: {result.kernel}")
    
    # Try to plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Main prediction plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Full prediction
        axes[0].scatter(X, y, c='black', s=15, alpha=0.7, label='Training data')
        axes[0].plot(X_test_orig, mean_orig, 'b-', lw=2, label='Mean prediction')
        axes[0].fill_between(X_test_orig, mean_orig - 2*std_orig, mean_orig + 2*std_orig,
                            alpha=0.3, color='blue', label='95% CI')
        axes[0].axvline(x=X.max(), color='red', linestyle='--', alpha=0.5, label='Training cutoff')
        axes[0].set_xlabel('Time (months)')
        axes[0].set_ylabel('Passengers')
        axes[0].set_title(f'Compositional Kernel Search Result\nBest Kernel: {result.kernel}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        pred_train, _ = gp.predict(X_norm)
        residuals = y_norm - pred_train
        axes[1].scatter(X, residuals * y_std, c='black', s=15, alpha=0.7)
        axes[1].axhline(y=0, color='red', linestyle='--')
        axes[1].set_xlabel('Time (months)')
        axes[1].set_ylabel('Residual')
        axes[1].set_title('Residuals')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/kernel_search_results.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved to: /mnt/user-data/outputs/kernel_search_results.png")
        
        # Search history plot
        fig2, ax = plt.subplots(figsize=(12, 6))
        depths = list(range(len(searcher.search_history)))
        bics = [h[3] for h in searcher.search_history]
        labels = [h[2][:30] + '...' if len(h[2]) > 30 else h[2] for h in searcher.search_history]
        
        ax.plot(depths, bics, 'bo-', markersize=10)
        for i, (d, b, l) in enumerate(zip(depths, bics, labels)):
            ax.annotate(l, (d, b), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=8, rotation=15)
        ax.set_xlabel('Search Depth')
        ax.set_ylabel('BIC (lower is better)')
        ax.set_title('Kernel Search Progress')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/search_progress.png', dpi=150, bbox_inches='tight')
        print("Search progress plot saved to: /mnt/user-data/outputs/search_progress.png")
        
    except Exception as e:
        print(f"\nCould not create plots: {e}")
    
    return result, gp


if __name__ == "__main__":
    result, gp = demo()
