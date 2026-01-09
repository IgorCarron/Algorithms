"""
Comprehensive Examples for Compositional Kernel Search

This script demonstrates the kernel search algorithm on multiple datasets
including the classic examples from the paper (Mauna Loa CO2, Airline passengers).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compositional_kernel_search import (
    CompositionalKernelSearch, GaussianProcess, 
    SquaredExponential, Periodic, Linear, RationalQuadratic,
    SumKernel, ProductKernel, decompose_posterior
)


def example_1_simple_periodic():
    """Example 1: Simple periodic data with trend."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Periodic Data with Linear Trend")
    print("="*70)
    
    # Generate data: linear trend + periodic + noise
    np.random.seed(42)
    X = np.linspace(0, 4*np.pi, 80)
    y = 0.5 * X + 2 * np.sin(X) + 0.3 * np.random.randn(len(X))
    
    # Normalize
    X_norm = (X - X.mean()) / X.std()
    y_norm = (y - y.mean()) / y.std()
    
    # Search
    searcher = CompositionalKernelSearch(max_depth=3, n_restarts=2, verbose=True)
    result = searcher.search(X_norm, y_norm)
    
    # Fit GP
    gp = GaussianProcess(result.kernel, noise_variance=result.noise_var)
    gp.fit(X_norm, y_norm)
    
    # Predict
    X_test = np.linspace(X_norm.min() - 0.5, X_norm.max() + 1, 200)
    mean, var = gp.predict(X_test)
    std = np.sqrt(var)
    
    # Denormalize
    X_test_orig = X_test * X.std() + X.mean()
    mean_orig = mean * y.std() + y.mean()
    std_orig = std * y.std()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(X, y, c='black', s=20, alpha=0.7, label='Data')
    ax.plot(X_test_orig, mean_orig, 'b-', lw=2, label='Mean')
    ax.fill_between(X_test_orig, mean_orig - 2*std_orig, mean_orig + 2*std_orig,
                    alpha=0.3, color='blue', label='95% CI')
    ax.axvline(x=X.max(), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Example 1: Simple Periodic + Trend\nDiscovered Kernel: {result.kernel}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def example_2_airline_passengers():
    """Example 2: Airline passengers with trend and growing seasonality."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Airline Passengers (Trend + Seasonality)")
    print("="*70)
    
    # Generate airline-like data
    np.random.seed(123)
    months = np.arange(144)  # 12 years
    
    # Components
    trend = 100 + 2.5 * months
    seasonality = 25 * np.sin(2 * np.pi * months / 12)
    # Growing amplitude
    growing_seasonality = seasonality * (1 + 0.008 * months)
    noise = 8 * np.random.randn(len(months))
    
    y = trend + growing_seasonality + noise
    
    # Normalize
    X_norm = (months - months.mean()) / months.std()
    y_norm = (y - y.mean()) / y.std()
    
    # Search
    searcher = CompositionalKernelSearch(max_depth=4, n_restarts=3, verbose=True)
    result = searcher.search(X_norm, y_norm)
    
    # Fit GP
    gp = GaussianProcess(result.kernel, noise_variance=result.noise_var)
    gp.fit(X_norm, y_norm)
    
    # Predict (including extrapolation)
    X_test = np.linspace(X_norm.min() - 0.2, X_norm.max() + 0.5, 250)
    mean, var = gp.predict(X_test)
    std = np.sqrt(var)
    
    # Denormalize
    X_test_orig = X_test * months.std() + months.mean()
    mean_orig = mean * y.std() + y.mean()
    std_orig = std * y.std()
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Main plot
    axes[0].scatter(months, y, c='black', s=15, alpha=0.7, label='Data')
    axes[0].plot(X_test_orig, mean_orig, 'b-', lw=2, label='Prediction')
    axes[0].fill_between(X_test_orig, mean_orig - 2*std_orig, mean_orig + 2*std_orig,
                         alpha=0.3, color='blue', label='95% CI')
    axes[0].axvline(x=months.max(), color='red', linestyle='--', alpha=0.5, label='Training cutoff')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Passengers')
    axes[0].set_title(f'Airline Passengers\nDiscovered: {result.kernel}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    pred, _ = gp.predict(X_norm)
    residuals = y_norm - pred
    axes[1].scatter(months, residuals * y.std(), c='black', s=15, alpha=0.7)
    axes[1].axhline(y=0, color='red', linestyle='--')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def example_3_mauna_loa_co2():
    """Example 3: Mauna Loa CO2 style data."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Mauna Loa CO2 (Long-term + Seasonal + Medium-term)")
    print("="*70)
    
    # Generate Mauna Loa-like data
    np.random.seed(456)
    t = np.arange(400)  # Monthly data
    
    # Long-term trend (approximately quadratic)
    trend = 310 + 0.1 * t + 0.0003 * t**2
    
    # Annual cycle
    seasonal = 3 * np.sin(2 * np.pi * t / 12)
    
    # Medium-term variations
    medium = 1.5 * np.sin(2 * np.pi * t / 50)
    
    noise = 0.3 * np.random.randn(len(t))
    
    y = trend + seasonal + medium + noise
    
    # Use only part for training (to show extrapolation)
    train_idx = t < 350
    X_train, y_train = t[train_idx], y[train_idx]
    
    # Normalize
    X_norm = (X_train - X_train.mean()) / X_train.std()
    y_norm = (y_train - y_train.mean()) / y_train.std()
    
    # Search
    searcher = CompositionalKernelSearch(max_depth=4, n_restarts=3, verbose=True)
    result = searcher.search(X_norm, y_norm)
    
    # Fit GP
    gp = GaussianProcess(result.kernel, noise_variance=result.noise_var)
    gp.fit(X_norm, y_norm)
    
    # Predict over full range
    X_test = np.linspace((t.min() - X_train.mean()) / X_train.std(),
                         (t.max() - X_train.mean()) / X_train.std(), 300)
    mean, var = gp.predict(X_test)
    std = np.sqrt(var)
    
    # Denormalize
    X_test_orig = X_test * X_train.std() + X_train.mean()
    mean_orig = mean * y_train.std() + y_train.mean()
    std_orig = std * y_train.std()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.scatter(X_train, y_train, c='black', s=10, alpha=0.5, label='Training data')
    ax.scatter(t[~train_idx], y[~train_idx], c='red', s=10, alpha=0.5, label='Test data')
    ax.plot(X_test_orig, mean_orig, 'b-', lw=2, label='Prediction')
    ax.fill_between(X_test_orig, mean_orig - 2*std_orig, mean_orig + 2*std_orig,
                    alpha=0.3, color='blue', label='95% CI')
    ax.axvline(x=X_train.max(), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('CO2 (ppm)')
    ax.set_title(f'Mauna Loa CO2 Style Data\nDiscovered: {result.kernel}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def example_4_locally_periodic():
    """Example 4: Locally periodic (SE × Per)."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Locally Periodic Data (Decaying Oscillations)")
    print("="*70)
    
    # Generate locally periodic data
    np.random.seed(789)
    X = np.linspace(0, 10, 150)
    
    # Locally periodic: periodic modulated by SE envelope
    y = np.sin(2 * np.pi * X) * np.exp(-0.15 * (X - 5)**2)
    y += 0.1 * np.random.randn(len(X))
    
    # Normalize
    X_norm = (X - X.mean()) / X.std()
    y_norm = (y - y.mean()) / y.std()
    
    # Search
    searcher = CompositionalKernelSearch(max_depth=3, n_restarts=3, verbose=True)
    result = searcher.search(X_norm, y_norm)
    
    # Fit GP
    gp = GaussianProcess(result.kernel, noise_variance=result.noise_var)
    gp.fit(X_norm, y_norm)
    
    # Predict
    X_test = np.linspace(X_norm.min() - 0.3, X_norm.max() + 0.5, 200)
    mean, var = gp.predict(X_test)
    std = np.sqrt(var)
    
    # Denormalize
    X_test_orig = X_test * X.std() + X.mean()
    mean_orig = mean * y.std() + y.mean()
    std_orig = std * y.std()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(X, y, c='black', s=20, alpha=0.7, label='Data')
    ax.plot(X_test_orig, mean_orig, 'b-', lw=2, label='Prediction')
    ax.fill_between(X_test_orig, mean_orig - 2*std_orig, mean_orig + 2*std_orig,
                    alpha=0.3, color='blue', label='95% CI')
    ax.axvline(x=X.max(), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Locally Periodic Data\nDiscovered: {result.kernel}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def example_5_multiscale():
    """Example 5: Multi-scale data (good for RQ kernel)."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Multi-scale Variation")
    print("="*70)
    
    # Generate multi-scale data
    np.random.seed(321)
    X = np.linspace(0, 10, 200)
    
    # Multiple scales
    y = np.sin(0.5 * X) + 0.5 * np.sin(2 * X) + 0.2 * np.sin(5 * X)
    y += 0.1 * np.random.randn(len(X))
    
    # Normalize
    X_norm = (X - X.mean()) / X.std()
    y_norm = (y - y.mean()) / y.std()
    
    # Search
    searcher = CompositionalKernelSearch(max_depth=4, n_restarts=3, verbose=True, use_rq=True)
    result = searcher.search(X_norm, y_norm)
    
    # Fit GP
    gp = GaussianProcess(result.kernel, noise_variance=result.noise_var)
    gp.fit(X_norm, y_norm)
    
    # Predict
    X_test = np.linspace(X_norm.min() - 0.2, X_norm.max() + 0.3, 250)
    mean, var = gp.predict(X_test)
    std = np.sqrt(var)
    
    # Denormalize
    X_test_orig = X_test * X.std() + X.mean()
    mean_orig = mean * y.std() + y.mean()
    std_orig = std * y.std()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(X, y, c='black', s=10, alpha=0.5, label='Data')
    ax.plot(X_test_orig, mean_orig, 'b-', lw=2, label='Prediction')
    ax.fill_between(X_test_orig, mean_orig - 2*std_orig, mean_orig + 2*std_orig,
                    alpha=0.3, color='blue', label='95% CI')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Multi-scale Data\nDiscovered: {result.kernel}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def example_6_decomposition():
    """Example 6: Show posterior decomposition."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Posterior Decomposition into Components")
    print("="*70)
    
    # Generate data with clear components
    np.random.seed(999)
    X = np.linspace(0, 10, 100)
    
    # Clear linear + periodic components
    linear_comp = 0.3 * X
    periodic_comp = np.sin(2 * np.pi * X / 2)
    noise = 0.15 * np.random.randn(len(X))
    
    y = linear_comp + periodic_comp + noise
    
    # Normalize
    X_norm = (X - X.mean()) / X.std()
    y_norm = (y - y.mean()) / y.std()
    
    # Create known kernel structure: Lin + Per
    kernel = SumKernel(Linear(0), Periodic(0))
    
    # Fit GP
    gp = GaussianProcess(kernel, noise_variance=0.1)
    gp.fit(X_norm, y_norm)
    gp.optimize(n_restarts=5)
    
    print(f"Fitted kernel: {gp.kernel}")
    
    # Get decomposition
    X_test = np.linspace(X_norm.min(), X_norm.max(), 150)
    components = decompose_posterior(gp, X_test)
    
    # Denormalize
    X_test_orig = X_test * X.std() + X.mean()
    
    # Full prediction
    mean_full, var_full = gp.predict(X_test)
    
    # Plot decomposition
    fig, axes = plt.subplots(len(components) + 2, 1, figsize=(12, 3*(len(components) + 2)))
    
    # Full model
    axes[0].scatter(X, y, c='black', s=20, alpha=0.7)
    mean_orig = mean_full * y.std() + y.mean()
    std_orig = np.sqrt(var_full) * y.std()
    axes[0].plot(X_test_orig, mean_orig, 'b-', lw=2)
    axes[0].fill_between(X_test_orig, mean_orig - 2*std_orig, mean_orig + 2*std_orig, alpha=0.3)
    axes[0].set_title(f'Full Model: {gp.kernel}')
    axes[0].set_ylabel('Y')
    axes[0].grid(True, alpha=0.3)
    
    # Components
    for i, (name, (comp_mean, comp_std)) in enumerate(components.items()):
        comp_mean_orig = comp_mean * y.std()
        comp_std_orig = comp_std * y.std()
        
        axes[i+1].plot(X_test_orig, comp_mean_orig, 'b-', lw=2)
        axes[i+1].fill_between(X_test_orig, 
                               comp_mean_orig - 2*comp_std_orig,
                               comp_mean_orig + 2*comp_std_orig, alpha=0.3)
        axes[i+1].set_title(f'Component: {name}')
        axes[i+1].set_ylabel('Y')
        axes[i+1].grid(True, alpha=0.3)
    
    # Residuals
    pred, _ = gp.predict(X_norm)
    residuals = (y_norm - pred) * y.std()
    axes[-1].scatter(X, residuals, c='black', s=20, alpha=0.7)
    axes[-1].axhline(y=0, color='red', linestyle='--')
    axes[-1].set_title('Residuals')
    axes[-1].set_xlabel('X')
    axes[-1].set_ylabel('Residual')
    axes[-1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_summary_figure():
    """Create a summary figure showing different kernel behaviors."""
    print("\n" + "="*70)
    print("Creating Summary of Base Kernels")
    print("="*70)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    X = np.linspace(-3, 3, 100)
    
    # Base kernels
    kernels_info = [
        (SquaredExponential(lengthscale=1.0), 'SE (Squared Exponential)', 'Local smoothness'),
        (Periodic(period=2.0, lengthscale=0.5), 'Periodic', 'Repeating patterns'),
        (Linear(), 'Linear', 'Linear functions'),
        (RationalQuadratic(alpha=1.0), 'Rational Quadratic', 'Multi-scale variation'),
    ]
    
    # Composite kernels
    composite_info = [
        (SumKernel(Linear(), SquaredExponential()), 'Lin + SE', 'Linear trend + local deviation'),
        (ProductKernel(Periodic(), SquaredExponential()), 'Per × SE', 'Locally periodic'),
        (ProductKernel(Linear(), Periodic()), 'Lin × Per', 'Growing amplitude'),
        (SumKernel(Periodic(), Linear()), 'Per + Lin', 'Periodic with trend'),
    ]
    
    np.random.seed(42)
    
    # Plot base kernels
    for i, (kernel, name, desc) in enumerate(kernels_info):
        gp = GaussianProcess(kernel, noise_variance=0.01)
        
        # Sample from prior
        K = kernel(X.reshape(-1, 1), X.reshape(-1, 1)) + 1e-6 * np.eye(len(X))
        L = np.linalg.cholesky(K)
        
        for _ in range(3):
            sample = L @ np.random.randn(len(X))
            axes[0, i].plot(X, sample, alpha=0.7)
        
        axes[0, i].set_title(f'{name}\n{desc}')
        axes[0, i].set_xlabel('x')
        axes[0, i].grid(True, alpha=0.3)
    
    # Plot composite kernels
    for i, (kernel, name, desc) in enumerate(composite_info):
        K = kernel(X.reshape(-1, 1), X.reshape(-1, 1)) + 1e-6 * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
            for _ in range(3):
                sample = L @ np.random.randn(len(X))
                axes[1, i].plot(X, sample, alpha=0.7)
        except:
            axes[1, i].text(0.5, 0.5, 'Sampling failed', ha='center', va='center')
        
        axes[1, i].set_title(f'{name}\n{desc}')
        axes[1, i].set_xlabel('x')
        axes[1, i].grid(True, alpha=0.3)
    
    axes[0, 0].set_ylabel('Base Kernels\nf(x)')
    axes[1, 0].set_ylabel('Composite Kernels\nf(x)')
    
    plt.suptitle('Kernel Functions: Base and Composite', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    """Run all examples and save figures."""
    print("\n" + "#"*70)
    print(" COMPOSITIONAL KERNEL SEARCH - COMPREHENSIVE EXAMPLES")
    print(" Implementation of Duvenaud et al., ICML 2013")
    print("#"*70)
    
    # Summary figure
    fig_summary = create_summary_figure()
    fig_summary.savefig('/mnt/user-data/outputs/kernel_summary.png', dpi=150, bbox_inches='tight')
    print("\nSaved: kernel_summary.png")
    
    # Example 1
    fig1 = example_1_simple_periodic()
    fig1.savefig('/mnt/user-data/outputs/example1_periodic_trend.png', dpi=150, bbox_inches='tight')
    print("Saved: example1_periodic_trend.png")
    plt.close(fig1)
    
    # Example 2
    fig2 = example_2_airline_passengers()
    fig2.savefig('/mnt/user-data/outputs/example2_airline.png', dpi=150, bbox_inches='tight')
    print("Saved: example2_airline.png")
    plt.close(fig2)
    
    # Example 3
    fig3 = example_3_mauna_loa_co2()
    fig3.savefig('/mnt/user-data/outputs/example3_mauna_loa.png', dpi=150, bbox_inches='tight')
    print("Saved: example3_mauna_loa.png")
    plt.close(fig3)
    
    # Example 4
    fig4 = example_4_locally_periodic()
    fig4.savefig('/mnt/user-data/outputs/example4_locally_periodic.png', dpi=150, bbox_inches='tight')
    print("Saved: example4_locally_periodic.png")
    plt.close(fig4)
    
    # Example 5
    fig5 = example_5_multiscale()
    fig5.savefig('/mnt/user-data/outputs/example5_multiscale.png', dpi=150, bbox_inches='tight')
    print("Saved: example5_multiscale.png")
    plt.close(fig5)
    
    # Example 6
    fig6 = example_6_decomposition()
    fig6.savefig('/mnt/user-data/outputs/example6_decomposition.png', dpi=150, bbox_inches='tight')
    print("Saved: example6_decomposition.png")
    plt.close(fig6)
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
