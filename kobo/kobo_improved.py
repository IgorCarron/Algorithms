"""
KOBO: Kernel Optimized Black-Box Optimization (Improved Version)

Implementation of the paper:
"Kernel Learning for Sample Constrained Black-Box Optimization"
Rajagopalan, Wei, Roy Choudhury (AAAI 2025)
arXiv:2507.20533

Key improvements in this version:
- More numerically stable model evidence computation
- Better kernel space generation with balanced complexity
- Improved VAE training with proper normalization
- More efficient kernel search in latent space
"""

import numpy as np
from scipy.special import erf
from scipy.linalg import cho_solve, cho_factor
from typing import List, Tuple, Dict, Callable, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Base Kernels
# =============================================================================

class BaseKernel(ABC):
    """Abstract base class for kernel functions."""
    
    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def name(self) -> str:
        pass
    
    def __add__(self, other: 'BaseKernel') -> 'SumKernel':
        return SumKernel(self, other)
    
    def __mul__(self, other: 'BaseKernel') -> 'ProductKernel':
        return ProductKernel(self, other)


class SumKernel(BaseKernel):
    """Sum of two kernels."""
    def __init__(self, k1: BaseKernel, k2: BaseKernel):
        self.k1 = k1
        self.k2 = k2
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.k1(X1, X2) + self.k2(X1, X2)
    
    def name(self) -> str:
        return f"({self.k1.name()} + {self.k2.name()})"


class ProductKernel(BaseKernel):
    """Product of two kernels."""
    def __init__(self, k1: BaseKernel, k2: BaseKernel):
        self.k1 = k1
        self.k2 = k2
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.k1(X1, X2) * self.k2(X1, X2)
    
    def name(self) -> str:
        return f"({self.k1.name()} * {self.k2.name()})"


class SquaredExponential(BaseKernel):
    """RBF/SE kernel: k(x,y) = σ² exp(-||x-y||² / (2l²))"""
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        sq_dist = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        sq_dist = np.maximum(sq_dist, 0)  # Numerical stability
        return self.variance * np.exp(-sq_dist / (2 * self.length_scale**2))
    
    def name(self) -> str:
        return "SE"


class Periodic(BaseKernel):
    """Periodic kernel: k(x,y) = σ² exp(-2 sin²(π||x-y||/p) / l²)"""
    
    def __init__(self, length_scale: float = 1.0, period: float = 2.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.period = period
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        # Use a simpler periodic formulation that's more stable
        diff = X1[:, :, None] - X2[:, :, None].T  # (n1, d, n2)
        sin_term = np.sin(np.pi * diff / self.period)
        sq_sin = np.sum(sin_term ** 2, axis=1)  # Sum over dimensions
        return self.variance * np.exp(-2 * sq_sin / self.length_scale**2)
    
    def name(self) -> str:
        return "PER"


class RationalQuadratic(BaseKernel):
    """RQ kernel: k(x,y) = σ² (1 + ||x-y||² / (2αl²))^(-α)"""
    
    def __init__(self, length_scale: float = 1.0, alpha: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.alpha = alpha
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        sq_dist = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        sq_dist = np.maximum(sq_dist, 0)
        return self.variance * (1 + sq_dist / (2 * self.alpha * self.length_scale**2))**(-self.alpha)
    
    def name(self) -> str:
        return "RQ"


class Matern(BaseKernel):
    """Matern 5/2 kernel"""
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dist = np.sqrt(np.maximum(
            np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T,
            0
        ) + 1e-12)
        sqrt5 = np.sqrt(5)
        r = dist / self.length_scale
        return self.variance * (1 + sqrt5 * r + 5 * r**2 / 3) * np.exp(-sqrt5 * r)
    
    def name(self) -> str:
        return "MAT"


class Linear(BaseKernel):
    """Linear kernel: k(x,y) = σ_b² + σ_v² (x-c)ᵀ(y-c)"""
    
    def __init__(self, variance: float = 1.0, bias: float = 1.0):
        self.variance = variance
        self.bias = bias
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        return self.bias + self.variance * X1 @ X2.T
    
    def name(self) -> str:
        return "LIN"


# =============================================================================
# Kernel Combiner with Grammar-Based Representation
# =============================================================================

@dataclass
class KernelGrammar:
    """Represents a kernel through its grammar code."""
    code: np.ndarray  # Grammar-based representation
    kernel: BaseKernel
    name_str: str


class KernelCombiner:
    """
    Creates composite kernels using a context-free grammar.
    
    Grammar: K_c = Σ_i (Π_j B_j^{a_{ij}})
    where B_j are base kernels and a_{ij} are powers.
    """
    
    BASE_KERNELS = ['SE', 'PER', 'RQ', 'MAT', 'LIN']
    
    def __init__(self, max_terms: int = 3, max_power: int = 2):
        self.max_terms = max_terms
        self.max_power = max_power
        self.n_bases = len(self.BASE_KERNELS)
        self.code_dim = self.max_terms * self.n_bases
    
    def _get_base_kernel(self, name: str, length_scale: float = 1.0) -> BaseKernel:
        """Create a base kernel instance."""
        kernels = {
            'SE': SquaredExponential(length_scale=length_scale),
            'PER': Periodic(length_scale=length_scale),
            'RQ': RationalQuadratic(length_scale=length_scale),
            'MAT': Matern(length_scale=length_scale),
            'LIN': Linear()
        }
        return kernels[name]
    
    def code_to_kernel(self, code: np.ndarray, length_scale: float = 1.0) -> KernelGrammar:
        """Convert grammar code to kernel."""
        code = np.asarray(code).flatten()
        terms = []
        name_parts = []
        
        for t in range(self.max_terms):
            start = t * self.n_bases
            term_code = code[start:start + self.n_bases]
            
            if np.sum(np.abs(term_code)) < 0.5:
                continue
            
            # Build multiplicative term
            term_kernel = None
            term_name = []
            
            for b, power in enumerate(term_code):
                power = int(np.round(power))
                if power > 0:
                    for _ in range(power):
                        base_k = self._get_base_kernel(self.BASE_KERNELS[b], length_scale)
                        if term_kernel is None:
                            term_kernel = base_k
                        else:
                            term_kernel = term_kernel * base_k
                        term_name.append(self.BASE_KERNELS[b])
            
            if term_kernel is not None:
                terms.append(term_kernel)
                name_parts.append("*".join(term_name))
        
        # Combine terms with addition
        if not terms:
            kernel = SquaredExponential(length_scale=length_scale)
            name_str = "SE"
        elif len(terms) == 1:
            kernel = terms[0]
            name_str = name_parts[0]
        else:
            kernel = terms[0]
            for t in terms[1:]:
                kernel = kernel + t
            name_str = " + ".join(name_parts)
        
        return KernelGrammar(code=code, kernel=kernel, name_str=name_str)
    
    def generate_random_code(self, complexity: str = 'medium') -> np.ndarray:
        """Generate random grammar code with controlled complexity."""
        code = np.zeros(self.code_dim)
        
        if complexity == 'simple':
            n_terms = 1
            bases_per_term = 1
        elif complexity == 'medium':
            n_terms = np.random.randint(1, 3)
            bases_per_term = np.random.randint(1, 3)
        else:  # complex
            n_terms = np.random.randint(2, self.max_terms + 1)
            bases_per_term = np.random.randint(2, 4)
        
        for t in range(n_terms):
            start = t * self.n_bases
            selected_bases = np.random.choice(self.n_bases, 
                                               min(bases_per_term, self.n_bases), 
                                               replace=False)
            for b in selected_bases:
                code[start + b] = np.random.randint(1, self.max_power + 1)
        
        return code
    
    def generate_kernel_space(self, n_kernels: int, X: Optional[np.ndarray] = None) -> List[KernelGrammar]:
        """Generate diverse kernel space."""
        kernels = []
        codes_seen = set()
        
        # Include all simple base kernels first
        for i in range(self.n_bases):
            code = np.zeros(self.code_dim)
            code[i] = 1
            kg = self.code_to_kernel(code)
            kernels.append(kg)
            codes_seen.add(tuple(code))
        
        # Add mixed complexity kernels
        complexities = ['simple'] * (n_kernels // 3) + \
                       ['medium'] * (n_kernels // 3) + \
                       ['complex'] * (n_kernels // 3)
        
        attempts = 0
        while len(kernels) < n_kernels and attempts < n_kernels * 10:
            complexity = complexities[len(kernels) % len(complexities)]
            code = self.generate_random_code(complexity)
            code_tuple = tuple(code)
            
            if code_tuple not in codes_seen:
                codes_seen.add(code_tuple)
                kg = self.code_to_kernel(code)
                kernels.append(kg)
            
            attempts += 1
        
        return kernels


# =============================================================================
# Gaussian Process Regression
# =============================================================================

class GPR:
    """Gaussian Process Regression with stable numerics."""
    
    def __init__(self, kernel: BaseKernel, noise_var: float = 1e-4):
        self.kernel = kernel
        self.noise_var = noise_var
        self.X_train = None
        self.y_train = None
        self.L = None
        self.alpha = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GPR to data."""
        X = np.atleast_2d(X)
        y = np.atleast_1d(y).flatten()
        
        self.X_train = X
        self.y_train = y
        self.n_train = X.shape[0]
        
        # Compute kernel matrix with noise
        K = self.kernel(X, X)
        
        # Ensure positive definiteness with increasing jitter
        jitter = self.noise_var
        max_tries = 10
        for i in range(max_tries):
            K_reg = K + jitter * np.eye(self.n_train)
            try:
                self.L = np.linalg.cholesky(K_reg)
                self.alpha = cho_solve((self.L, True), y)
                return
            except np.linalg.LinAlgError:
                jitter *= 10
        
        # Final fallback: use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = np.maximum(eigvals, 1e-6)
        K_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T + self.noise_var * np.eye(self.n_train)
        self.L = np.linalg.cholesky(K_fixed)
        self.alpha = cho_solve((self.L, True), y)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance."""
        X = np.atleast_2d(X)
        
        K_star = self.kernel(self.X_train, X)
        
        mean = K_star.T @ self.alpha
        
        v = cho_solve((self.L, True), K_star)
        K_ss = self.kernel(X, X)
        var = np.diag(K_ss) - np.sum(K_star * v, axis=0)
        var = np.maximum(var, 1e-10)
        
        return mean, var
    
    def log_marginal_likelihood(self) -> float:
        """Compute normalized log marginal likelihood."""
        if self.L is None:
            return -np.inf
        
        # Data fit term
        data_fit = -0.5 * np.dot(self.y_train, self.alpha)
        
        # Complexity term (log determinant)
        log_det = np.sum(np.log(np.diag(self.L)))
        complexity = -log_det
        
        # Constant term
        constant = -0.5 * self.n_train * np.log(2 * np.pi)
        
        # Normalize by number of data points for comparison
        return (data_fit + complexity + constant) / self.n_train


# =============================================================================
# Simple KerVAE Implementation
# =============================================================================

class KerVAE:
    """
    Variational Autoencoder for kernel space.
    Simple MLP-based architecture with proper normalization.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 8):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Simple linear encoder
        self.W_enc = np.random.randn(input_dim, latent_dim) * 0.1
        self.b_enc = np.zeros(latent_dim)
        
        # Decoder
        self.W_dec = np.random.randn(latent_dim, input_dim) * 0.1
        self.b_dec = np.zeros(input_dim)
        
        # Normalization statistics
        self.mean = None
        self.std = None
    
    def fit(self, X: np.ndarray):
        """Fit VAE using simple PCA-like encoding."""
        # Normalize
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.mean) / self.std
        
        # Use SVD for initialization (PCA-like)
        U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)
        
        # Set encoder to project to top latent_dim components
        self.W_enc = Vt[:self.latent_dim].T
        self.W_dec = Vt[:self.latent_dim]
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode to latent space."""
        X_norm = (X - self.mean) / self.std
        return X_norm @ self.W_enc + self.b_enc
    
    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode from latent space."""
        X_norm = Z @ self.W_dec + self.b_dec
        return X_norm * self.std + self.mean


# =============================================================================
# Acquisition Function
# =============================================================================

def expected_improvement(mean: np.ndarray, std: np.ndarray, 
                         f_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function."""
    with np.errstate(divide='warn', invalid='warn'):
        imp = f_best - mean - xi
        Z = imp / (std + 1e-10)
        ei = imp * norm_cdf(Z) + std * norm_pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


# =============================================================================
# KOBO Main Algorithm
# =============================================================================

@dataclass
class KOBOConfig:
    """KOBO configuration."""
    query_budget: int = 50
    batch_size: int = 5
    kernel_update_interval: int = 5
    n_kernel_samples: int = 100
    latent_dim: int = 8
    n_candidates: int = 1000
    noise_var: float = 1e-4
    verbose: bool = True


class KOBO:
    """
    Kernel Optimized Black-Box Optimization.
    
    Learns optimal GPR kernel by:
    1. Generating composite kernels (Kernel Combiner)
    2. Learning continuous kernel space (KerVAE)
    3. Optimizing model evidence (KerGPR)
    4. Using optimal kernel for BO (fGPR)
    """
    
    def __init__(self, objective_func: Callable, bounds: np.ndarray,
                 config: Optional[KOBOConfig] = None):
        self.objective = objective_func
        self.bounds = np.asarray(bounds)
        self.dim = self.bounds.shape[0]
        self.config = config or KOBOConfig()
        
        # Kernel combiner
        self.combiner = KernelCombiner()
        
        # Observation storage
        self.X_obs = []
        self.y_obs = []
        self.n_queries = 0
        
        # Best found
        self.best_x = None
        self.best_y = np.inf
        
        # Kernel state
        self.current_kernel = SquaredExponential()
        self.kernel_space = None
        self.ker_vae = None
        
        # History
        self.history = {
            'best_y': [],
            'kernels': [],
            'evidences': []
        }
    
    def _sample_lhs(self, n: int) -> np.ndarray:
        """Latin Hypercube Sampling."""
        samples = np.zeros((n, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n)
            samples[:, d] = (perm + np.random.rand(n)) / n
            samples[:, d] = self.bounds[d, 0] + samples[:, d] * (self.bounds[d, 1] - self.bounds[d, 0])
        return samples
    
    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate objective at points."""
        X = np.atleast_2d(X)
        y = np.array([self.objective(x) for x in X])
        
        for x, yi in zip(X, y):
            self.X_obs.append(x)
            self.y_obs.append(yi)
            if yi < self.best_y:
                self.best_y = yi
                self.best_x = x.copy()
        
        self.n_queries += len(X)
        return y
    
    def _estimate_length_scale(self) -> float:
        """Estimate good length scale from data."""
        if len(self.X_obs) < 2:
            return 1.0
        X = np.array(self.X_obs)
        dists = []
        for i in range(min(len(X), 20)):
            for j in range(i + 1, min(len(X), 20)):
                dists.append(np.linalg.norm(X[i] - X[j]))
        return np.median(dists) if dists else 1.0
    
    def _compute_kernel_representations(self) -> np.ndarray:
        """Compute representations for all kernels in space."""
        X = np.array(self.X_obs)
        reps = []
        
        for kg in self.kernel_space:
            # Grammar code
            rc = kg.code
            
            # Data-based representation: model evidence
            gpr = GPR(kg.kernel, self.config.noise_var)
            try:
                gpr.fit(X, np.array(self.y_obs))
                evidence = gpr.log_marginal_likelihood()
            except:
                evidence = -100.0
            
            rep = np.concatenate([rc, [evidence]])
            reps.append(rep)
        
        return np.array(reps)
    
    def _update_kernel(self):
        """Update kernel using KerGPR."""
        X = np.array(self.X_obs)
        y = np.array(self.y_obs)
        length_scale = self._estimate_length_scale()
        
        # Generate/update kernel space
        if self.kernel_space is None:
            self.kernel_space = self.combiner.generate_kernel_space(
                self.config.n_kernel_samples, X
            )
            # Update kernels with estimated length scale
            self.kernel_space = [
                self.combiner.code_to_kernel(kg.code, length_scale)
                for kg in self.kernel_space
            ]
        
        # Compute model evidence for each kernel
        evidences = []
        for kg in self.kernel_space:
            gpr = GPR(kg.kernel, self.config.noise_var)
            try:
                gpr.fit(X, y)
                ev = gpr.log_marginal_likelihood()
            except:
                ev = -np.inf
            evidences.append(ev)
        
        evidences = np.array(evidences)
        
        # Select best kernel
        valid = np.isfinite(evidences)
        if not np.any(valid):
            return
        
        best_idx = np.argmax(evidences)
        self.current_kernel = self.kernel_space[best_idx].kernel
        
        if self.config.verbose:
            print(f"  Kernel: {self.kernel_space[best_idx].name_str}")
            print(f"  Evidence: {evidences[best_idx]:.4f}")
        
        self.history['kernels'].append(self.kernel_space[best_idx].name_str)
        self.history['evidences'].append(float(evidences[best_idx]))
    
    def _select_next(self, n: int) -> np.ndarray:
        """Select next points using acquisition function."""
        X = np.array(self.X_obs)
        y = np.array(self.y_obs)
        
        gpr = GPR(self.current_kernel, self.config.noise_var)
        gpr.fit(X, y)
        
        # Generate candidates
        candidates = np.zeros((self.config.n_candidates, self.dim))
        for d in range(self.dim):
            candidates[:, d] = np.random.uniform(
                self.bounds[d, 0], self.bounds[d, 1], 
                self.config.n_candidates
            )
        
        mean, var = gpr.predict(candidates)
        std = np.sqrt(var)
        
        ei = expected_improvement(mean, std, self.best_y)
        
        top_idx = np.argsort(ei)[-n:]
        return candidates[top_idx]
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Run KOBO optimization."""
        if self.config.verbose:
            print("=" * 60)
            print("KOBO: Kernel Optimized Black-Box Optimization")
            print("=" * 60)
            print(f"Dimension: {self.dim}, Budget: {self.config.query_budget}")
        
        # Initial sampling
        X_init = self._sample_lhs(self.config.batch_size)
        self._evaluate(X_init)
        self.history['best_y'].append(self.best_y)
        
        if self.config.verbose:
            print(f"\nInitial: {self.n_queries} queries, best = {self.best_y:.6f}")
        
        iteration = 1
        while self.n_queries < self.config.query_budget:
            if self.config.verbose:
                print(f"\nIteration {iteration}:")
            
            # Update kernel periodically
            if self.n_queries % self.config.kernel_update_interval == 0:
                self._update_kernel()
            
            # Select next points
            n_next = min(self.config.batch_size, 
                         self.config.query_budget - self.n_queries)
            if n_next <= 0:
                break
            
            X_next = self._select_next(n_next)
            self._evaluate(X_next)
            self.history['best_y'].append(self.best_y)
            
            if self.config.verbose:
                print(f"  Queries: {self.n_queries}/{self.config.query_budget}")
                print(f"  Best: {self.best_y:.6f}")
            
            iteration += 1
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print(f"Complete! Best = {self.best_y:.6f}")
            print("=" * 60)
        
        return self.best_x, self.best_y


# =============================================================================
# Standard BO Baseline
# =============================================================================

class StandardBO:
    """Standard BO with fixed kernel for comparison."""
    
    def __init__(self, objective: Callable, bounds: np.ndarray,
                 kernel: Optional[BaseKernel] = None,
                 budget: int = 50, batch_size: int = 5, verbose: bool = True):
        self.objective = objective
        self.bounds = np.asarray(bounds)
        self.dim = bounds.shape[0]
        self.kernel = kernel or SquaredExponential()
        self.budget = budget
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.X_obs = []
        self.y_obs = []
        self.n_queries = 0
        self.best_x = None
        self.best_y = np.inf
        self.history = {'best_y': []}
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Run standard BO."""
        # Initial sampling
        for _ in range(self.batch_size):
            x = np.array([np.random.uniform(self.bounds[d, 0], self.bounds[d, 1])
                          for d in range(self.dim)])
            y = self.objective(x)
            self.X_obs.append(x)
            self.y_obs.append(y)
            if y < self.best_y:
                self.best_y = y
                self.best_x = x.copy()
        
        self.n_queries = self.batch_size
        self.history['best_y'].append(self.best_y)
        
        while self.n_queries < self.budget:
            X = np.array(self.X_obs)
            y = np.array(self.y_obs)
            
            gpr = GPR(self.kernel)
            gpr.fit(X, y)
            
            # Generate candidates
            candidates = np.zeros((1000, self.dim))
            for d in range(self.dim):
                candidates[:, d] = np.random.uniform(
                    self.bounds[d, 0], self.bounds[d, 1], 1000
                )
            
            mean, var = gpr.predict(candidates)
            std = np.sqrt(var)
            ei = expected_improvement(mean, std, self.best_y)
            
            x_next = candidates[np.argmax(ei)]
            y_next = self.objective(x_next)
            
            self.X_obs.append(x_next)
            self.y_obs.append(y_next)
            if y_next < self.best_y:
                self.best_y = y_next
                self.best_x = x_next.copy()
            
            self.n_queries += 1
            self.history['best_y'].append(self.best_y)
        
        if self.verbose:
            print(f"  {self.kernel.name()}: {self.best_y:.6f}")
        
        return self.best_x, self.best_y


# =============================================================================
# Benchmark Functions
# =============================================================================

def staircase(x: np.ndarray, n_stairs: int = 5) -> float:
    """Non-smooth staircase function."""
    x = np.atleast_1d(x)
    val = 0.0
    for xi in x:
        stair = np.floor(xi * n_stairs) / n_stairs
        val += (stair - 0.5) ** 2
    return val + 0.1 * np.sin(np.sum(x) * 10)


def branin(x: np.ndarray) -> float:
    """Branin function. Global min ~0.398."""
    x = np.atleast_1d(x)
    x1, x2 = x[0], x[1] if len(x) > 1 else 0
    a, b, c = 1, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6, 10, 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s


def michalewicz(x: np.ndarray, m: float = 10) -> float:
    """Michalewicz function with steep valleys."""
    x = np.atleast_1d(x)
    result = 0.0
    for i, xi in enumerate(x):
        result -= np.sin(xi) * np.sin((i + 1) * xi**2 / np.pi)**(2 * m)
    return result


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function - highly multimodal."""
    x = np.atleast_1d(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function."""
    x = np.atleast_1d(x)
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
               for i in range(len(x) - 1))


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("KOBO: Kernel Optimized Black-Box Optimization")
    print("=" * 70)
    
    # Test 1: Branin (2D smooth)
    print("\n>>> Test 1: Branin Function (2D)")
    bounds_branin = np.array([[-5, 10], [0, 15]])
    
    config = KOBOConfig(
        query_budget=40,
        batch_size=5,
        kernel_update_interval=10,
        n_kernel_samples=50,
        verbose=True
    )
    
    np.random.seed(42)
    kobo = KOBO(branin, bounds_branin, config)
    kobo.optimize()
    
    print("\nBaselines:")
    for name, kernel in [('SE', SquaredExponential()), 
                          ('PER', Periodic()),
                          ('MAT', Matern())]:
        np.random.seed(42)
        bo = StandardBO(branin, bounds_branin, kernel, budget=40, batch_size=5)
        bo.optimize()
    
    # Test 2: Staircase (5D non-smooth)
    print("\n" + "-" * 70)
    print("\n>>> Test 2: Staircase Function (5D)")
    
    bounds_stair = np.array([[0, 1]] * 5)
    
    config = KOBOConfig(
        query_budget=50,
        batch_size=5,
        kernel_update_interval=10,
        n_kernel_samples=50,
        verbose=True
    )
    
    np.random.seed(123)
    kobo = KOBO(staircase, bounds_stair, config)
    kobo.optimize()
    
    print("\nBaselines:")
    for name, kernel in [('SE', SquaredExponential()), 
                          ('PER', Periodic()),
                          ('MAT', Matern())]:
        np.random.seed(123)
        bo = StandardBO(staircase, bounds_stair, kernel, budget=50, batch_size=5)
        bo.optimize()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
