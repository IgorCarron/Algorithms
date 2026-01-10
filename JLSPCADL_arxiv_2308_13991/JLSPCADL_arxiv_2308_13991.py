import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, classification_report
from scipy.linalg import eigh
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class JLSPCADL:
    """
    Johnson-Lindenstrauss Supervised PCA Discriminative Dictionary Learning
    
    This class implements the JLSPCADL algorithm which combines:
    1. Optimal projection dimension calculation using JL-lemma
    2. Modified Supervised PCA for maximum feature-label consistency
    3. Dictionary learning in the transformed space
    """
    
    def __init__(self, epsilon_range: Tuple[float, float] = (0.3, 0.4), 
                 K_factor: float = 2.0, tolerance: float = 1e-6):
        """
        Initialize JLSPCADL
        
        Args:
            epsilon_range: Range for optimal perturbation threshold selection
            K_factor: Factor to determine dictionary size (K = K_factor * p)
            tolerance: Convergence tolerance for dictionary learning
        """
        self.epsilon_range = epsilon_range
        self.K_factor = K_factor
        self.tolerance = tolerance
        self.optimal_p = None
        self.optimal_epsilon = None
        self.U = None  # Projection matrix
        self.D = None  # Dictionary
        self.medoids = None  # Class medoids
        
    def _compute_optimal_epsilon_p(self, N: int, max_features: int) -> Tuple[float, int]:
        """
        Compute optimal perturbation threshold and projection dimension using JL-lemma
        
        Args:
            N: Number of training samples
            max_features: Maximum possible projection dimension
            
        Returns:
            Tuple of (optimal_epsilon, optimal_p)
        """
        eps_values = np.linspace(self.epsilon_range[0], self.epsilon_range[1], 50)
        p_values = []
        derivatives = []
        
        for eps in eps_values:
            # JL-lemma projection dimension: p >= 12*log(N) / (eps^2 * (1.5 - eps))
            p_raw = 12 * np.log(N) / (eps**2 * (1.5 - eps))
            p = int(np.ceil(p_raw))
            # Constrain to be at most max_features
            p = min(p, max_features)
            p_values.append(p)
            
            # Compute derivative dp/deps
            if eps > 0.001 and eps < 0.999:
                derivative = 36 * np.log(N) * (eps - 1) / (eps**3 * (1.5 - eps)**2)
                derivatives.append(abs(derivative))
            else:
                derivatives.append(float('inf'))
        
        # Find epsilon where derivative is minimum (curve flattens)
        min_derivative_idx = np.argmin(derivatives)
        optimal_epsilon = eps_values[min_derivative_idx]
        optimal_p = p_values[min_derivative_idx]
        
        return optimal_epsilon, optimal_p
    
    def _modified_supervised_pca(self, Y: np.ndarray, H: np.ndarray, p: int) -> np.ndarray:
        """
        Modified Supervised PCA to compute projection matrix with maximum feature-label consistency
        
        Args:
            Y: Data matrix (d x N)
            H: Label matrix (C x N) - one-hot encoded
            p: Number of principal components (projection dimension)
            
        Returns:
            Projection matrix U (d x p)
        """
        # Compute label kernel matrix L = H^T * H
        L = H.T @ H
        
        # Compute Y * L * Y^T for HSIC criterion
        YL = Y @ L
        YLYT = YL @ Y.T
        
        # Add small regularization for numerical stability
        YLYT += 1e-8 * np.eye(YLYT.shape[0])
        
        # Find p largest eigenvalues and eigenvectors
        eigenvals, eigenvecs = eigh(YLYT)
        
        # Sort in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Select top p eigenvectors as projection matrix
        U = eigenvecs[:, :p]
        
        return U
    
    def _k_svd_step(self, Z: np.ndarray, D: np.ndarray, X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single K-SVD step to update one dictionary atom
        
        Args:
            Z: Transformed data (p x N)
            D: Current dictionary (p x K)
            X: Current coefficients (K x N)
            k: Index of atom to update
            
        Returns:
            Updated dictionary D and coefficients X
        """
        # Find samples that use atom k
        omega_k = np.where(X[k, :] != 0)[0]
        
        if len(omega_k) == 0:
            return D, X
        
        # Compute residual without atom k
        D_k = D.copy()
        D_k[:, k] = 0
        E_k = Z[:, omega_k] - D_k @ X[:, omega_k]
        
        # SVD to update atom k
        if E_k.shape[1] > 0:
            U, s, Vt = np.linalg.svd(E_k, full_matrices=False)
            if len(s) > 0:
                D[:, k] = U[:, 0]
                X[k, omega_k] = s[0] * Vt[0, :]
        
        return D, X
    
    def _sparse_coding(self, Z: np.ndarray, D: np.ndarray, lambda_reg: float = 0.1) -> np.ndarray:
        """
        Sparse coding step using iterative soft thresholding
        
        Args:
            Z: Transformed data (p x N)
            D: Dictionary (p x K)
            lambda_reg: Regularization parameter
            
        Returns:
            Sparse coefficient matrix X (K x N)
        """
        K, N = D.shape[1], Z.shape[1]
        X = np.random.randn(K, N) * 0.1
        
        # Compute step size
        L = np.linalg.norm(D.T @ D, 2)
        step_size = 1.0 / L if L > 0 else 1.0
        
        for _ in range(50):  # Fixed number of iterations
            # Gradient step
            residual = Z - D @ X
            gradient = -D.T @ residual
            X_new = X - step_size * gradient
            
            # Soft thresholding
            threshold = lambda_reg * step_size
            X_new = np.sign(X_new) * np.maximum(np.abs(X_new) - threshold, 0)
            
            X = X_new
        
        return X
    
    def _compute_medoids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute medoids of sparse coefficients for each class
        
        Args:
            X: Coefficient matrix (K x N)
            labels: Class labels (N,)
            
        Returns:
            Medoids matrix (K x C)
        """
        unique_labels = np.unique(labels)
        C = len(unique_labels)
        K = X.shape[0]
        medoids = np.zeros((K, C))
        
        for i, label in enumerate(unique_labels):
            class_mask = labels == label
            class_coeffs = X[:, class_mask]
            
            if class_coeffs.shape[1] > 0:
                # Compute pairwise distances and find medoid
                distances = np.zeros(class_coeffs.shape[1])
                for j in range(class_coeffs.shape[1]):
                    diff = class_coeffs - class_coeffs[:, [j]]
                    distances[j] = np.sum(np.linalg.norm(diff, axis=0))
                
                medoid_idx = np.argmin(distances)
                medoids[:, i] = class_coeffs[:, medoid_idx]
        
        return medoids
    
    def fit(self, Y: np.ndarray, labels: np.ndarray, max_iter: int = 20) -> 'JLSPCADL':
        """
        Fit the JLSPCADL model
        
        Args:
            Y: Training data matrix (N x d)
            labels: Training labels (N,)
            max_iter: Maximum iterations for dictionary learning
            
        Returns:
            Self
        """
        # Transpose to match paper notation (d x N)
        Y = Y.T
        N, d = Y.shape[1], Y.shape[0]
        
        # Standardize data
        scaler = StandardScaler()
        Y_scaled = scaler.fit_transform(Y.T).T
        self.scaler = scaler
        
        # Create one-hot encoded label matrix
        unique_labels = np.unique(labels)
        C = len(unique_labels)
        H = np.zeros((C, N))
        for i, label in enumerate(unique_labels):
            H[i, labels == label] = 1
        
        # Step 1: Compute optimal epsilon and p using JL-lemma
        # Constrain p to be at most min(d-1, N-1)
        max_features = min(d - 1, N - 1)
        self.optimal_epsilon, self.optimal_p = self._compute_optimal_epsilon_p(N, max_features)
        print(f"Optimal epsilon: {self.optimal_epsilon:.3f}, Optimal p: {self.optimal_p}")
        
        # Ensure p doesn't exceed data dimensionality
        p = min(self.optimal_p, max_features)
        
        # Step 2: Compute projection matrix using Modified Supervised PCA
        self.U = self._modified_supervised_pca(Y_scaled, H, p)
        
        # Step 3: Transform data
        Z = self.U.T @ Y_scaled
        
        # Step 4: Dictionary learning in transformed space
        K = int(self.K_factor * p)  # Dictionary size
        
        # Initialize dictionary
        self.D = np.random.randn(p, K)
        # Normalize dictionary atoms
        norms = np.linalg.norm(self.D, axis=0, keepdims=True)
        norms[norms == 0] = 1
        self.D = self.D / norms
        
        # Alternating optimization
        prev_error = float('inf')
        
        for iteration in range(max_iter):
            # Sparse coding step
            X = self._sparse_coding(Z, self.D)
            
            # Dictionary update step (simplified K-SVD)
            for k in range(K):
                self.D, X = self._k_svd_step(Z, self.D, X, k)
            
            # Normalize dictionary atoms
            norms = np.linalg.norm(self.D, axis=0, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            self.D = self.D / norms
            
            # Check convergence
            error = np.linalg.norm(Z - self.D @ X, 'fro')
            if abs(prev_error - error) < self.tolerance:
                print(f"Converged at iteration {iteration + 1}")
                break
            prev_error = error
        
        # Compute class medoids
        self.medoids = self._compute_medoids(X, labels)
        
        return self
    
    def predict(self, Y_test: np.ndarray, tau: float = 0.35) -> np.ndarray:
        """
        Predict labels for test data
        
        Args:
            Y_test: Test data matrix (N_test x d)
            tau: Weight parameter for classification rule
            
        Returns:
            Predicted labels
        """
        # Transform test data
        Y_test_scaled = self.scaler.transform(Y_test).T
        Z_test = self.U.T @ Y_test_scaled
        
        # Sparse coding for test samples
        X_test = self._sparse_coding(Z_test, self.D)
        
        # Classification using equation (4.14)
        predictions = []
        for i in range(X_test.shape[1]):
            x_q = X_test[:, [i]]
            z_q = Z_test[:, [i]]
            
            min_cost = float('inf')
            best_class = 0
            
            for c in range(self.medoids.shape[1]):
                # Reconstruction error
                recon_error = np.linalg.norm(z_q - self.D @ x_q)**2
                
                # Distance to class medoid
                medoid_dist = np.linalg.norm(x_q.flatten() - self.medoids[:, c])**2
                
                # Total cost
                cost = recon_error + tau * medoid_dist
                
                if cost < min_cost:
                    min_cost = cost
                    best_class = c
            
            predictions.append(best_class)
        
        return np.array(predictions)

def generate_sample_data(n_samples: int = 1000, n_features: int = 50, n_classes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample classification data for demonstration
    """
    X, y = make_classification(n_samples=n_samples, 
                             n_features=n_features,
                             n_informative=n_features//2,
                             n_redundant=0,
                             n_classes=n_classes,
                             random_state=42)
    return X, y

def plot_jl_lemma_analysis(N_values: np.ndarray):
    """
    Plot JL-lemma projection dimension analysis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Projection dimension vs epsilon for different N
    eps_values = np.linspace(0.1, 0.9, 100)
    
    for N in [100, 500, 1000, 5000]:
        p_values = []
        for eps in eps_values:
            if eps < 1.5 and eps > 0:
                p = 12 * np.log(N) / (eps**2 * (1.5 - eps))
                p_values.append(p)
            else:
                p_values.append(np.nan)
        ax1.plot(eps_values, p_values, label=f'N={N}')
    
    ax1.set_xlabel('Perturbation Threshold (ε)')
    ax1.set_ylabel('Projection Dimension (p)')
    ax1.set_title('JL-Lemma: Projection Dimension vs Epsilon')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.2, 0.8)
    ax1.set_ylim(0, 2000)
    
    # Plot 2: Derivative analysis
    eps_values = np.linspace(0.25, 0.8, 100)
    N = 1000
    
    derivatives = []
    for eps in eps_values:
        if eps < 1.5 and eps > 0:
            derivative = abs(36 * np.log(N) * (eps - 1) / (eps**3 * (1.5 - eps)**2))
            derivatives.append(derivative)
        else:
            derivatives.append(np.nan)
    
    ax2.plot(eps_values, derivatives, 'r-', linewidth=2)
    ax2.axvspan(0.3, 0.4, alpha=0.3, color='green', label='Optimal ε range')
    ax2.set_xlabel('Perturbation Threshold (ε)')
    ax2.set_ylabel('|dp/dε|')
    ax2.set_title('Derivative Analysis for Optimal ε Selection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_projection_comparison(X_original, X_pca, X_jlspcadl, y):
    """
    Compare PCA vs JLSPCADL projections
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original data (first 2 dimensions)
    scatter1 = axes[0].scatter(X_original[:, 0], X_original[:, 1], c=y, cmap='tab10', alpha=0.7)
    axes[0].set_title('Original Data (first 2 dims)')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # PCA projection
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
    axes[1].set_title('Standard PCA Projection')
    axes[1].set_xlabel('PC 1')
    axes[1].set_ylabel('PC 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    # JLSPCADL projection
    scatter3 = axes[2].scatter(X_jlspcadl[:, 0], X_jlspcadl[:, 1], c=y, cmap='tab10', alpha=0.7)
    axes[2].set_title('JLSPCADL Projection')
    axes[2].set_xlabel('Component 1')
    axes[2].set_ylabel('Component 2')
    plt.colorbar(scatter3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

def plot_performance_comparison(results):
    """
    Plot performance comparison between different methods
    """
    plt.figure(figsize=(10, 6))
    
    methods = list(results.keys())
    accuracies = [results[method]['accuracy'] for method in methods]
    
    bars = plt.bar(methods, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.ylabel('Classification Accuracy')
    plt.title('Performance Comparison: JLSPCADL vs Baseline Methods')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("JLSPCADL Implementation Demo")
    print("=" * 40)
    
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data(n_samples=800, n_features=30, n_classes=5)
    
    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Data dimensionality: {X_train.shape[1]}, Number of classes: {len(np.unique(y))}")
    
    # Plot 1: JL-lemma analysis
    print("\nGenerating JL-lemma analysis plots...")
    N_values = np.array([100, 500, 1000, 5000])
    plot_jl_lemma_analysis(N_values)
    
    # Fit JLSPCADL model
    print("\nFitting JLSPCADL model...")
    jlspcadl = JLSPCADL(epsilon_range=(0.3, 0.4), K_factor=2.0)
    jlspcadl.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_jlspcadl = jlspcadl.predict(X_test)
    acc_jlspcadl = accuracy_score(y_test, y_pred_jlspcadl)
    
    # Compare with standard PCA + Nearest Centroid
    print("\nComparing with baseline methods...")
    
    # Standard PCA - ensure n_components doesn't exceed data limits
    actual_p = jlspcadl.U.shape[1]
    max_components = min(X_train.shape[0] - 1, X_train.shape[1] - 1)
    n_components_pca = min(actual_p, max_components)
    
    pca = PCA(n_components=n_components_pca)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    clf_pca = NearestCentroid()
    clf_pca.fit(X_train_pca, y_train)
    y_pred_pca = clf_pca.predict(X_test_pca)
    acc_pca = accuracy_score(y_test, y_pred_pca)
    
    # Nearest Centroid on original data
    clf_original = NearestCentroid()
    clf_original.fit(X_train, y_train)
    y_pred_original = clf_original.predict(X_test)
    acc_original = accuracy_score(y_test, y_pred_original)
    
    # Plot 2: Projection comparison (2D visualization)
    print("\nGenerating projection comparison...")
    # For visualization, use first 2 components
    pca_2d = PCA(n_components=2)
    X_train_pca_2d = pca_2d.fit_transform(X_train)
    
    # Get 2D projection from JLSPCADL
    U_2d = jlspcadl.U[:, :2]  # First 2 components
    X_train_scaled = jlspcadl.scaler.transform(X_train)
    X_train_jlspcadl_2d = X_train_scaled @ U_2d
    
    plot_projection_comparison(X_train, X_train_pca_2d, X_train_jlspcadl_2d, y_train)
    
    # Plot 3: Performance comparison
    print("\nGenerating performance comparison...")
    results = {
        'Original Data\n+ Nearest Centroid': {'accuracy': acc_original},
        'PCA\n+ Nearest Centroid': {'accuracy': acc_pca},
        'JLSPCADL': {'accuracy': acc_jlspcadl}
    }
    
    plot_performance_comparison(results)
    
    # Print results
    print("\nResults Summary:")
    print("=" * 40)
    print(f"JLSPCADL Parameters:")
    print(f"  - Optimal ε: {jlspcadl.optimal_epsilon:.3f}")
    print(f"  - Optimal p: {jlspcadl.optimal_p}")
    print(f"  - Actual p used: {jlspcadl.U.shape[1]}")
    print(f"  - Dictionary size K: {jlspcadl.D.shape[1]}")
    
    print(f"\nClassification Accuracies:")
    print(f"  - Original data + Nearest Centroid: {acc_original:.3f}")
    print(f"  - PCA + Nearest Centroid: {acc_pca:.3f}")
    print(f"  - JLSPCADL: {acc_jlspcadl:.3f}")
    
    print(f"\nImprovement over PCA: {acc_jlspcadl - acc_pca:.3f}")
    print(f"Improvement over original: {acc_jlspcadl - acc_original:.3f}")
    
    print("\nJLSPCADL Demo completed successfully!")