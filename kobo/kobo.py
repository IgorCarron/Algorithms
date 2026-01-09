"""
KOBO: Kernel Optimized Black-Box Optimization

Implementation of the paper:
"Kernel Learning for Sample Constrained Black-Box Optimization"
Rajagopalan, Wei, Roy Choudhury (AAAI 2025)
arXiv:2507.20533

The algorithm learns an optimal GPR kernel by:
1. Creating composite kernels via a context-free grammar (Kernel Combiner)
2. Learning a continuous latent space of kernels (KerVAE)
3. Optimizing model evidence in this latent space (KerGPR)
4. Using the optimal kernel for black-box optimization (fGPR)
"""

import numpy as np
from scipy.special import erf
from typing import List, Tuple, Dict, Callable, Optional
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
        """Compute kernel matrix K[i,j] = k(X1[i], X2[j])"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        pass


class SquaredExponential(BaseKernel):
    """Squared Exponential (RBF) kernel: k(x,y) = exp(-||x-y||^2 / (2*l^2))"""
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return self.variance * np.exp(-dists / (2 * self.length_scale**2))
    
    def name(self) -> str:
        return "SE"


class Periodic(BaseKernel):
    """Periodic kernel: k(x,y) = exp(-2*sin^2(pi*||x-y||/p) / l^2)"""
    
    def __init__(self, length_scale: float = 1.0, period: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.period = period
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = np.sqrt(np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T + 1e-10)
        return self.variance * np.exp(-2 * np.sin(np.pi * dists / self.period)**2 / self.length_scale**2)
    
    def name(self) -> str:
        return "PER"


class RationalQuadratic(BaseKernel):
    """Rational Quadratic kernel: k(x,y) = (1 + ||x-y||^2 / (2*alpha*l^2))^(-alpha)"""
    
    def __init__(self, length_scale: float = 1.0, alpha: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.alpha = alpha
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return self.variance * (1 + dists / (2 * self.alpha * self.length_scale**2))**(-self.alpha)
    
    def name(self) -> str:
        return "RQ"


class Matern(BaseKernel):
    """Matern kernel with nu=2.5"""
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = np.sqrt(np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T + 1e-10)
        sqrt5 = np.sqrt(5)
        r = dists / self.length_scale
        return self.variance * (1 + sqrt5 * r + 5 * r**2 / 3) * np.exp(-sqrt5 * r)
    
    def name(self) -> str:
        return "MAT"


class Linear(BaseKernel):
    """Linear kernel: k(x,y) = sigma_b^2 + sigma_v^2 * (x - c) * (y - c)"""
    
    def __init__(self, variance: float = 1.0, bias: float = 0.0, center: float = 0.0):
        self.variance = variance
        self.bias = bias
        self.center = center
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        return self.bias + self.variance * (X1 - self.center) @ (X2 - self.center).T
    
    def name(self) -> str:
        return "LIN"


# =============================================================================
# Composite Kernel Operations
# =============================================================================

class CompositeKernel(BaseKernel):
    """Composite kernel formed by combining base kernels."""
    
    def __init__(self, kernels: List[BaseKernel], operations: List[str], grammar_code: np.ndarray):
        """
        Args:
            kernels: List of base kernels
            operations: List of operations ('add' or 'mul') between kernels
            grammar_code: The grammar-based representation vector
        """
        self.kernels = kernels
        self.operations = operations
        self.grammar_code = grammar_code
        self._name = self._build_name()
    
    def _build_name(self) -> str:
        if len(self.kernels) == 0:
            return "Empty"
        if len(self.kernels) == 1:
            return self.kernels[0].name()
        
        result = self.kernels[0].name()
        for i, (k, op) in enumerate(zip(self.kernels[1:], self.operations)):
            if op == 'mul':
                result = f"({result} * {k.name()})"
            else:
                result = f"({result} + {k.name()})"
        return result
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if len(self.kernels) == 0:
            return np.ones((X1.shape[0], X2.shape[0]))
        
        # Start with first kernel
        result = self.kernels[0](X1, X2)
        
        # Apply operations sequentially
        for k, op in zip(self.kernels[1:], self.operations):
            K = k(X1, X2)
            if op == 'mul':
                result = result * K
            else:  # add
                result = result + K
        
        return result
    
    def name(self) -> str:
        return self._name


# =============================================================================
# Kernel Combiner
# =============================================================================

class KernelCombiner:
    """
    Creates composite kernels from base kernels using a context-free grammar.
    
    The grammar-based representation follows:
    k_C = A^a1 * B^b1 * C^c1 * D^d1 * E^e1
        + A^a2 * B^b2 * C^c2 * D^d2 * E^e2
        + ...
    
    Where A,B,C,D,E are base kernels (SE, PER, RQ, MAT, LIN)
    """
    
    def __init__(self, base_kernel_classes: Optional[List] = None, 
                 max_terms: int = 3, max_power: int = 2):
        """
        Args:
            base_kernel_classes: List of base kernel classes
            max_terms: Maximum number of additive terms
            max_power: Maximum power for each base kernel
        """
        if base_kernel_classes is None:
            self.base_kernel_classes = [
                SquaredExponential, Periodic, RationalQuadratic, Matern, Linear
            ]
        else:
            self.base_kernel_classes = base_kernel_classes
        
        self.base_names = ['SE', 'PER', 'RQ', 'MAT', 'LIN']
        self.max_terms = max_terms
        self.max_power = max_power
        self.n_bases = len(self.base_kernel_classes)
        self.code_length = self.max_terms * self.n_bases
    
    def generate_random_code(self) -> np.ndarray:
        """Generate a random grammar code."""
        code = np.zeros(self.code_length)
        n_terms = np.random.randint(1, self.max_terms + 1)
        
        for term_idx in range(n_terms):
            # Randomly select which bases to use in this term
            n_bases_in_term = np.random.randint(1, self.n_bases + 1)
            selected_bases = np.random.choice(self.n_bases, n_bases_in_term, replace=False)
            
            for base_idx in selected_bases:
                power = np.random.randint(1, self.max_power + 1)
                code[term_idx * self.n_bases + base_idx] = power
        
        return code
    
    def code_to_kernel(self, code: np.ndarray) -> CompositeKernel:
        """Convert a grammar code to a composite kernel."""
        terms = []  # List of (kernel, term_index) for each multiplicative term
        
        for term_idx in range(self.max_terms):
            term_start = term_idx * self.n_bases
            term_code = code[term_start:term_start + self.n_bases]
            
            # Check if this term has any non-zero powers
            if np.sum(np.abs(term_code)) < 0.01:
                continue
            
            # Build multiplicative term
            term_kernels = []
            for base_idx, power in enumerate(term_code):
                if power > 0.01:
                    # Add kernel(s) based on power
                    for _ in range(int(np.round(power))):
                        kernel = self.base_kernel_classes[base_idx]()
                        term_kernels.append(kernel)
            
            if term_kernels:
                terms.append(term_kernels)
        
        # Build composite kernel
        if not terms:
            # Default to SE kernel if code is empty
            return CompositeKernel([SquaredExponential()], [], code)
        
        all_kernels = []
        all_operations = []
        
        for term_idx, term_kernels in enumerate(terms):
            for k_idx, kernel in enumerate(term_kernels):
                all_kernels.append(kernel)
                if k_idx > 0:
                    all_operations.append('mul')
            
            # Add 'add' operation between terms (except before first term)
            if term_idx < len(terms) - 1:
                all_operations.append('add')
        
        return CompositeKernel(all_kernels, all_operations, code)
    
    def generate_kernel_space(self, n_kernels: int) -> List[CompositeKernel]:
        """Generate a diverse set of composite kernels."""
        kernels = []
        codes_seen = set()
        
        while len(kernels) < n_kernels:
            code = self.generate_random_code()
            code_tuple = tuple(code.round(2))
            
            if code_tuple not in codes_seen:
                codes_seen.add(code_tuple)
                kernel = self.code_to_kernel(code)
                kernels.append(kernel)
        
        return kernels
    
    def compute_data_representation(self, kernel: CompositeKernel, 
                                     X: np.ndarray) -> np.ndarray:
        """
        Compute data-based representation: distances between composite 
        kernel's covariance matrix and each base kernel's covariance matrix.
        
        rd = ||M_C - M_b||_F for each base kernel b
        """
        M_C = kernel(X, X)
        
        rd = np.zeros(self.n_bases)
        for i, base_cls in enumerate(self.base_kernel_classes):
            base_kernel = base_cls()
            M_b = base_kernel(X, X)
            rd[i] = np.linalg.norm(M_C - M_b, 'fro')
        
        return rd
    
    def get_full_representation(self, kernel: CompositeKernel, 
                                 X: np.ndarray) -> np.ndarray:
        """Get combined grammar-based and data-based representation."""
        rc = kernel.grammar_code
        rd = self.compute_data_representation(kernel, X)
        return np.concatenate([rc, rd])


# =============================================================================
# Variational Autoencoder for Kernel Space (KerVAE)
# =============================================================================

class KerVAE:
    """
    Kernel Space Variational Autoencoder.
    
    Learns a continuous latent space Z from the discrete kernel space K.
    Uses a simple MLP-based VAE architecture.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 8, 
                 hidden_dims: List[int] = None, learning_rate: float = 1e-3):
        """
        Args:
            input_dim: Dimension of kernel representation (grammar + data)
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimization
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.lr = learning_rate
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize encoder and decoder weights."""
        np.random.seed(42)
        
        # Encoder weights
        self.encoder_weights = []
        self.encoder_biases = []
        
        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            W = np.random.randn(prev_dim, h_dim) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(h_dim)
            self.encoder_weights.append(W)
            self.encoder_biases.append(b)
            prev_dim = h_dim
        
        # Mean and log-variance layers
        self.W_mu = np.random.randn(prev_dim, self.latent_dim) * np.sqrt(2.0 / prev_dim)
        self.b_mu = np.zeros(self.latent_dim)
        self.W_logvar = np.random.randn(prev_dim, self.latent_dim) * np.sqrt(2.0 / prev_dim)
        self.b_logvar = np.zeros(self.latent_dim)
        
        # Decoder weights
        self.decoder_weights = []
        self.decoder_biases = []
        
        prev_dim = self.latent_dim
        for h_dim in reversed(self.hidden_dims):
            W = np.random.randn(prev_dim, h_dim) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(h_dim)
            self.decoder_weights.append(W)
            self.decoder_biases.append(b)
            prev_dim = h_dim
        
        # Output layer
        self.W_out = np.random.randn(prev_dim, self.input_dim) * np.sqrt(2.0 / prev_dim)
        self.b_out = np.zeros(self.input_dim)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode input to latent distribution parameters."""
        h = x
        for W, b in zip(self.encoder_weights, self.encoder_biases):
            h = self._relu(h @ W + b)
        
        mu = h @ self.W_mu + self.b_mu
        logvar = h @ self.W_logvar + self.b_logvar
        
        return mu, logvar
    
    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Reparameterization trick: z = mu + std * epsilon."""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vector to reconstruction."""
        h = z
        for W, b in zip(self.decoder_weights, self.decoder_biases):
            h = self._relu(h @ W + b)
        
        return h @ self.W_out + self.b_out
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss(self, x: np.ndarray, recon: np.ndarray, 
             mu: np.ndarray, logvar: np.ndarray) -> float:
        """Compute ELBO loss = reconstruction loss + KL divergence."""
        recon_loss = np.mean(np.sum((x - recon)**2, axis=-1))
        kl_loss = -0.5 * np.mean(np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=-1))
        return recon_loss + kl_loss
    
    def train(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32,
              verbose: bool = False):
        """Train VAE using simple gradient descent."""
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            total_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i:i+batch_size]
                
                # Forward pass
                recon, mu, logvar = self.forward(batch)
                loss = self.loss(batch, recon, mu, logvar)
                total_loss += loss
                n_batches += 1
                
                # Simple gradient update using finite differences
                self._update_weights(batch, loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")
    
    def _update_weights(self, batch: np.ndarray, current_loss: float):
        """Update weights using numerical gradients (simplified)."""
        eps = 1e-4
        
        # Update encoder weights
        for idx in range(len(self.encoder_weights)):
            grad_W = np.zeros_like(self.encoder_weights[idx])
            for i in range(min(5, self.encoder_weights[idx].shape[0])):
                for j in range(min(5, self.encoder_weights[idx].shape[1])):
                    self.encoder_weights[idx][i, j] += eps
                    recon, mu, logvar = self.forward(batch)
                    loss_plus = self.loss(batch, recon, mu, logvar)
                    self.encoder_weights[idx][i, j] -= eps
                    grad_W[i, j] = (loss_plus - current_loss) / eps
            
            self.encoder_weights[idx] -= self.lr * grad_W
        
        # Update decoder weights similarly (simplified for efficiency)
        for idx in range(len(self.decoder_weights)):
            grad_W = np.zeros_like(self.decoder_weights[idx])
            for i in range(min(5, self.decoder_weights[idx].shape[0])):
                for j in range(min(5, self.decoder_weights[idx].shape[1])):
                    self.decoder_weights[idx][i, j] += eps
                    recon, mu, logvar = self.forward(batch)
                    loss_plus = self.loss(batch, recon, mu, logvar)
                    self.decoder_weights[idx][i, j] -= eps
                    grad_W[i, j] = (loss_plus - current_loss) / eps
            
            self.decoder_weights[idx] -= self.lr * grad_W


# =============================================================================
# Gaussian Process Regression
# =============================================================================

class GPR:
    """
    Gaussian Process Regression.
    
    Used for both:
    - fGPR: Function optimization with learned kernel
    - KerGPR: Kernel optimization in latent space
    """
    
    def __init__(self, kernel: BaseKernel, noise_var: float = 1e-4):
        """
        Args:
            kernel: Kernel function
            noise_var: Observation noise variance
        """
        self.kernel = kernel
        self.noise_var = noise_var
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GPR to training data."""
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        
        self.X_train = X
        self.y_train = y
        
        K = self.kernel(X, X) + self.noise_var * np.eye(X.shape[0])
        
        # Add jitter for numerical stability
        K += 1e-6 * np.eye(K.shape[0])
        
        try:
            L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self.K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(K.shape[0])))
            self.L = L
        except np.linalg.LinAlgError:
            # Fallback to direct inverse
            self.K_inv = np.linalg.inv(K)
            self.alpha = self.K_inv @ y
            self.L = None
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at new points."""
        X = np.atleast_2d(X)
        
        K_star = self.kernel(self.X_train, X)
        K_star_star = self.kernel(X, X)
        
        mean = K_star.T @ self.alpha
        var = K_star_star - K_star.T @ self.K_inv @ K_star
        
        # Return diagonal of variance matrix
        var_diag = np.diag(var).clip(min=1e-10)
        
        return mean, var_diag
    
    def log_marginal_likelihood(self) -> float:
        """Compute log marginal likelihood (model evidence)."""
        if self.L is not None:
            log_det = 2 * np.sum(np.log(np.diag(self.L)))
        else:
            sign, log_det = np.linalg.slogdet(
                self.kernel(self.X_train, self.X_train) + self.noise_var * np.eye(self.X_train.shape[0])
            )
            log_det = sign * log_det
        
        n = len(self.y_train)
        data_fit = -0.5 * self.y_train @ self.alpha
        complexity = -0.5 * log_det
        constant = -0.5 * n * np.log(2 * np.pi)
        
        return data_fit + complexity + constant


# =============================================================================
# Acquisition Functions
# =============================================================================

def expected_improvement(gpr: GPR, X: np.ndarray, f_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function."""
    mean, var = gpr.predict(X)
    std = np.sqrt(var)
    
    with np.errstate(divide='warn'):
        imp = f_best - mean - xi
        Z = imp / std
        ei = imp * _norm_cdf(Z) + std * _norm_pdf(Z)
        ei[std < 1e-10] = 0.0
    
    return ei


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


# =============================================================================
# KOBO: Kernel Optimized Black-Box Optimization
# =============================================================================

@dataclass
class KOBOConfig:
    """Configuration for KOBO algorithm."""
    
    # Budget and batching
    query_budget: int = 50
    batch_size: int = 5
    kernel_update_interval: int = 5
    
    # Kernel space
    n_kernel_samples: int = 200
    latent_dim: int = 8
    
    # VAE training
    vae_epochs: int = 100
    vae_batch_size: int = 32
    
    # Optimization
    n_candidates: int = 1000
    noise_var: float = 1e-4
    
    # Verbosity
    verbose: bool = True


class KOBO:
    """
    Kernel Optimized Black-Box Optimization (KOBO).
    
    Main algorithm that combines:
    - Kernel Combiner: Generates composite kernels
    - KerVAE: Learns continuous kernel latent space
    - KerGPR: Optimizes kernel in latent space
    - fGPR: Optimizes objective function with learned kernel
    """
    
    def __init__(self, objective_func: Callable, bounds: np.ndarray, 
                 config: Optional[KOBOConfig] = None):
        """
        Args:
            objective_func: Black-box function to optimize (minimize)
            bounds: Array of shape (dim, 2) with [lower, upper] bounds
            config: KOBO configuration
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.dim = self.bounds.shape[0]
        self.config = config or KOBOConfig()
        
        # Initialize kernel combiner
        self.kernel_combiner = KernelCombiner()
        
        # Initialize data storage
        self.X_observed = []
        self.y_observed = []
        self.queries_used = 0
        
        # Best found so far
        self.best_x = None
        self.best_y = float('inf')
        
        # Current kernel
        self.current_kernel = SquaredExponential()
        
        # KerVAE and kernel space
        self.ker_vae = None
        self.kernel_space = None
        self.kernel_representations = None
        
        # History for analysis
        self.history = {
            'best_y': [],
            'kernel_names': [],
            'model_evidence': []
        }
    
    def _sample_initial_points(self, n_points: int) -> np.ndarray:
        """Sample initial points using Latin Hypercube Sampling."""
        samples = np.zeros((n_points, self.dim))
        
        for d in range(self.dim):
            perms = np.random.permutation(n_points)
            samples[:, d] = (perms + np.random.rand(n_points)) / n_points
            samples[:, d] = self.bounds[d, 0] + samples[:, d] * (self.bounds[d, 1] - self.bounds[d, 0])
        
        return samples
    
    def _evaluate_points(self, X: np.ndarray) -> np.ndarray:
        """Evaluate objective function at given points."""
        X = np.atleast_2d(X)
        y = np.array([self.objective_func(x) for x in X])
        
        # Update best
        min_idx = np.argmin(y)
        if y[min_idx] < self.best_y:
            self.best_y = y[min_idx]
            self.best_x = X[min_idx].copy()
        
        # Store observations
        for x, yi in zip(X, y):
            self.X_observed.append(x)
            self.y_observed.append(yi)
        
        self.queries_used += len(X)
        return y
    
    def _initialize_kernel_space(self):
        """Initialize kernel space and train KerVAE."""
        if self.config.verbose:
            print("Initializing kernel space...")
        
        # Generate diverse kernels
        self.kernel_space = self.kernel_combiner.generate_kernel_space(
            self.config.n_kernel_samples
        )
        
        # Compute representations
        X = np.array(self.X_observed)
        self.kernel_representations = []
        
        for kernel in self.kernel_space:
            rep = self.kernel_combiner.get_full_representation(kernel, X)
            self.kernel_representations.append(rep)
        
        self.kernel_representations = np.array(self.kernel_representations)
        
        # Train KerVAE
        input_dim = self.kernel_representations.shape[1]
        self.ker_vae = KerVAE(
            input_dim=input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=[64, 32],
            learning_rate=1e-3
        )
        
        self.ker_vae.train(
            self.kernel_representations,
            epochs=self.config.vae_epochs,
            batch_size=self.config.vae_batch_size,
            verbose=False
        )
        
        if self.config.verbose:
            print(f"  Kernel space initialized with {len(self.kernel_space)} kernels")
    
    def _update_kernel_via_kergpr(self):
        """Use KerGPR to find optimal kernel in latent space."""
        if self.ker_vae is None:
            self._initialize_kernel_space()
        else:
            # Re-compute data representations with new observations
            X = np.array(self.X_observed)
            self.kernel_representations = []
            for kernel in self.kernel_space:
                rep = self.kernel_combiner.get_full_representation(kernel, X)
                self.kernel_representations.append(rep)
            self.kernel_representations = np.array(self.kernel_representations)
            
            # Retrain VAE
            self.ker_vae.train(
                self.kernel_representations,
                epochs=self.config.vae_epochs // 2,
                batch_size=self.config.vae_batch_size,
                verbose=False
            )
        
        # Encode kernels to latent space
        mu, _ = self.ker_vae.encode(self.kernel_representations)
        
        # Compute model evidence for each kernel
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        model_evidences = []
        for kernel in self.kernel_space:
            gpr = GPR(kernel, noise_var=self.config.noise_var)
            try:
                gpr.fit(X, y)
                evidence = gpr.log_marginal_likelihood()
            except:
                evidence = -np.inf
            model_evidences.append(evidence)
        
        model_evidences = np.array(model_evidences)
        
        # Use KerGPR to find optimal kernel in latent space
        # Simple approach: use GPR on latent representations
        valid_mask = np.isfinite(model_evidences)
        if not np.any(valid_mask):
            return  # Keep current kernel
        
        ker_gpr = GPR(SquaredExponential(length_scale=1.0), noise_var=0.01)
        ker_gpr.fit(mu[valid_mask], model_evidences[valid_mask])
        
        # Find best in current kernel space
        pred_evidence, _ = ker_gpr.predict(mu)
        best_idx = np.argmax(pred_evidence)
        
        self.current_kernel = self.kernel_space[best_idx]
        
        if self.config.verbose:
            print(f"  Updated kernel: {self.current_kernel.name()}")
            print(f"  Model evidence: {model_evidences[best_idx]:.4f}")
        
        self.history['kernel_names'].append(self.current_kernel.name())
        self.history['model_evidence'].append(float(model_evidences[best_idx]))
    
    def _select_next_points(self, n_points: int) -> np.ndarray:
        """Select next points to evaluate using acquisition function."""
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Fit GPR with current kernel
        gpr = GPR(self.current_kernel, noise_var=self.config.noise_var)
        gpr.fit(X, y)
        
        # Generate candidate points
        candidates = np.zeros((self.config.n_candidates, self.dim))
        for d in range(self.dim):
            candidates[:, d] = np.random.uniform(
                self.bounds[d, 0], self.bounds[d, 1], self.config.n_candidates
            )
        
        # Compute acquisition function
        ei = expected_improvement(gpr, candidates, self.best_y)
        
        # Select top n_points
        top_indices = np.argsort(ei)[-n_points:]
        
        return candidates[top_indices]
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run KOBO optimization.
        
        Returns:
            best_x: Best found point
            best_y: Best found value
        """
        if self.config.verbose:
            print("=" * 60)
            print("KOBO: Kernel Optimized Black-Box Optimization")
            print("=" * 60)
            print(f"Input dimension: {self.dim}")
            print(f"Query budget: {self.config.query_budget}")
            print(f"Batch size: {self.config.batch_size}")
            print()
        
        # Initial sampling
        initial_points = self._sample_initial_points(self.config.batch_size)
        self._evaluate_points(initial_points)
        
        if self.config.verbose:
            print(f"Initial sampling: {self.queries_used} queries, best_y = {self.best_y:.6f}")
        
        self.history['best_y'].append(self.best_y)
        
        # Main optimization loop
        iteration = 1
        while self.queries_used < self.config.query_budget:
            if self.config.verbose:
                print(f"\nIteration {iteration}:")
            
            # Update kernel if interval reached
            if self.queries_used % self.config.kernel_update_interval == 0:
                self._update_kernel_via_kergpr()
            
            # Select and evaluate next points
            n_points = min(
                self.config.batch_size,
                self.config.query_budget - self.queries_used
            )
            
            if n_points <= 0:
                break
            
            next_points = self._select_next_points(n_points)
            self._evaluate_points(next_points)
            
            self.history['best_y'].append(self.best_y)
            
            if self.config.verbose:
                print(f"  Queries used: {self.queries_used}/{self.config.query_budget}")
                print(f"  Best y: {self.best_y:.6f}")
            
            iteration += 1
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Optimization Complete!")
            print(f"Best x: {self.best_x}")
            print(f"Best y: {self.best_y:.6f}")
            print(f"Total queries: {self.queries_used}")
            print("=" * 60)
        
        return self.best_x, self.best_y


# =============================================================================
# Baseline: Standard Bayesian Optimization
# =============================================================================

class StandardBO:
    """Standard Bayesian Optimization with fixed kernel (for comparison)."""
    
    def __init__(self, objective_func: Callable, bounds: np.ndarray,
                 kernel: Optional[BaseKernel] = None, 
                 query_budget: int = 50, batch_size: int = 5,
                 verbose: bool = True):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.dim = self.bounds.shape[0]
        self.kernel = kernel or SquaredExponential()
        self.query_budget = query_budget
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.X_observed = []
        self.y_observed = []
        self.queries_used = 0
        self.best_x = None
        self.best_y = float('inf')
        self.history = {'best_y': []}
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Run standard BO optimization."""
        if self.verbose:
            print(f"Standard BO with {self.kernel.name()} kernel")
        
        # Initial sampling
        for d in range(self.dim):
            x = np.random.uniform(self.bounds[d, 0], self.bounds[d, 1], self.batch_size)
            if d == 0:
                X_init = x.reshape(-1, 1)
            else:
                X_init = np.column_stack([X_init, x])
        
        if self.dim == 1:
            X_init = X_init.reshape(-1, 1)
        
        for x in X_init:
            y = self.objective_func(x)
            self.X_observed.append(x)
            self.y_observed.append(y)
            if y < self.best_y:
                self.best_y = y
                self.best_x = x.copy()
        
        self.queries_used = len(X_init)
        self.history['best_y'].append(self.best_y)
        
        # Main loop
        while self.queries_used < self.query_budget:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            
            gpr = GPR(self.kernel)
            gpr.fit(X, y)
            
            # Generate candidates
            candidates = np.zeros((1000, self.dim))
            for d in range(self.dim):
                candidates[:, d] = np.random.uniform(
                    self.bounds[d, 0], self.bounds[d, 1], 1000
                )
            
            ei = expected_improvement(gpr, candidates, self.best_y)
            best_idx = np.argmax(ei)
            next_x = candidates[best_idx]
            
            next_y = self.objective_func(next_x)
            self.X_observed.append(next_x)
            self.y_observed.append(next_y)
            
            if next_y < self.best_y:
                self.best_y = next_y
                self.best_x = next_x.copy()
            
            self.queries_used += 1
            self.history['best_y'].append(self.best_y)
        
        if self.verbose:
            print(f"  Best y: {self.best_y:.6f} after {self.queries_used} queries")
        
        return self.best_x, self.best_y


# =============================================================================
# Test Functions (Benchmarks)
# =============================================================================

def staircase_function(x: np.ndarray, n_stairs: int = 5) -> float:
    """
    Staircase function - exhibits non-smooth structure.
    Common in user satisfaction modeling.
    """
    x = np.atleast_1d(x)
    val = 0.0
    for i, xi in enumerate(x):
        stair = np.floor(xi * n_stairs) / n_stairs
        val += (stair - 0.5) ** 2
    return val + 0.1 * np.sin(np.sum(x) * 10)


def branin_function(x: np.ndarray) -> float:
    """
    Branin function - smooth benchmark.
    Input: 2D, bounds typically [[-5, 10], [0, 15]]
    Global minima: ~0.398
    """
    x = np.atleast_1d(x)
    x1, x2 = x[0], x[1] if len(x) > 1 else 0
    
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s


def michalewicz_function(x: np.ndarray, m: float = 10) -> float:
    """
    Michalewicz function - periodic benchmark with steep valleys.
    """
    x = np.atleast_1d(x)
    d = len(x)
    result = 0.0
    for i in range(d):
        result -= np.sin(x[i]) * np.sin((i + 1) * x[i]**2 / np.pi)**(2 * m)
    return result


def rastrigin_function(x: np.ndarray) -> float:
    """Rastrigin function - highly multimodal."""
    x = np.atleast_1d(x)
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock function - valley-shaped."""
    x = np.atleast_1d(x)
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
               for i in range(len(x) - 1))


# =============================================================================
# Main Demo
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("KOBO: Kernel Optimized Black-Box Optimization - Demo")
    print("=" * 70)
    
    # Test on Branin function (2D)
    print("\n>>> Test 1: Branin Function (2D, Smooth)")
    bounds = np.array([[-5, 10], [0, 15]])
    
    config = KOBOConfig(
        query_budget=30,
        batch_size=3,
        kernel_update_interval=6,
        n_kernel_samples=100,
        vae_epochs=50,
        verbose=True
    )
    
    np.random.seed(42)
    kobo = KOBO(branin_function, bounds, config)
    best_x, best_y = kobo.optimize()
    
    print("\n>>> Comparison with Standard BO (SE kernel):")
    np.random.seed(42)
    bo_se = StandardBO(branin_function, bounds, SquaredExponential(), 
                       query_budget=30, batch_size=3, verbose=True)
    bo_se.optimize()
    
    # Test on Staircase function (higher dimensional)
    print("\n" + "-" * 70)
    print("\n>>> Test 2: Staircase Function (5D, Non-smooth)")
    
    dim = 5
    bounds = np.array([[0, 1]] * dim)
    
    config = KOBOConfig(
        query_budget=40,
        batch_size=5,
        kernel_update_interval=10,
        n_kernel_samples=100,
        vae_epochs=50,
        verbose=True
    )
    
    np.random.seed(123)
    kobo = KOBO(staircase_function, bounds, config)
    best_x, best_y = kobo.optimize()
    
    print("\n>>> Comparison with Standard BO (SE kernel):")
    np.random.seed(123)
    bo_se = StandardBO(staircase_function, bounds, SquaredExponential(),
                       query_budget=40, batch_size=5, verbose=True)
    bo_se.optimize()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
