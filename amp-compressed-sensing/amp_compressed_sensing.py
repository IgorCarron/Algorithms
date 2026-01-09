"""
Approximate Message Passing (AMP) for Compressed Sensing

Implementation based on:
Krzakala, Mezard, Sausset, Sun, Zdeborova (2012)
"Probabilistic reconstruction in compressed sensing: algorithms, phase diagrams,
and threshold achieving matrices"
https://arxiv.org/abs/1109.4424

Also draws from:
- Donoho, Maleki, Montanari (2009) - AMP for compressed sensing
- Bayati, Montanari (2011) - State evolution analysis

The algorithm reconstructs a sparse signal x from measurements y = Ax + noise
where A is an M x N measurement matrix with M << N.
"""

import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt


# =============================================================================
# Denoising Functions (Proximal Operators)
# =============================================================================

def soft_threshold(r, tau):
    """
    Soft thresholding denoiser (L1 regularization).
    η(r, τ) = sign(r) * max(|r| - τ, 0)

    Args:
        r: noisy estimate (N,) array
        tau: threshold parameter

    Returns:
        Denoised estimate
    """
    return np.sign(r) * np.maximum(np.abs(r) - tau, 0)


def soft_threshold_derivative(r, tau):
    """
    Derivative of soft thresholding w.r.t. r.
    Used for Onsager correction term.
    """
    return (np.abs(r) > tau).astype(float)


def gauss_bernoulli_denoiser(r, sigma_sq, rho, sigma_x=1.0):
    """
    Bayesian optimal denoiser for Gauss-Bernoulli prior.

    Prior: x_i ~ (1-rho)*delta(0) + rho*N(0, sigma_x^2)

    This is the MMSE estimator: E[x | r] where r = x + N(0, sigma_sq)

    Args:
        r: noisy observation r = x + noise
        sigma_sq: noise variance in the effective channel
        rho: sparsity (fraction of non-zero components)
        sigma_x: std of non-zero components

    Returns:
        MMSE estimate of x given r
    """
    # Effective variance
    var_eff = sigma_sq + sigma_x**2

    # Likelihood ratio
    # P(r|x!=0) / P(r|x=0)
    log_ratio = -0.5 * np.log(var_eff / sigma_sq) + \
                0.5 * r**2 * (1/sigma_sq - 1/var_eff)

    # Posterior probability that x != 0
    # Using log-sum-exp for numerical stability
    log_prior_ratio = np.log(rho / (1 - rho + 1e-10) + 1e-10)
    log_odds = log_prior_ratio + log_ratio

    # Sigmoid for numerical stability
    prob_nonzero = 1 / (1 + np.exp(-np.clip(log_odds, -500, 500)))

    # Posterior mean of x given x != 0
    # E[x | x!=0, r] = r * sigma_x^2 / (sigma_sq + sigma_x^2)
    mean_nonzero = r * sigma_x**2 / var_eff

    # MMSE estimate: E[x|r] = P(x!=0|r) * E[x|x!=0,r]
    return prob_nonzero * mean_nonzero


def gauss_bernoulli_denoiser_derivative(r, sigma_sq, rho, sigma_x=1.0, eps=1e-6):
    """
    Numerical derivative of Gauss-Bernoulli denoiser.
    Used for Onsager correction.
    """
    f_plus = gauss_bernoulli_denoiser(r + eps, sigma_sq, rho, sigma_x)
    f_minus = gauss_bernoulli_denoiser(r - eps, sigma_sq, rho, sigma_x)
    return (f_plus - f_minus) / (2 * eps)


def bernoulli_denoiser(r, sigma_sq, rho, x_val=1.0):
    """
    Bayesian optimal denoiser for Bernoulli (binary sparse) prior.

    Prior: x_i ~ (1-rho)*delta(0) + rho*delta(x_val)

    Args:
        r: noisy observation
        sigma_sq: noise variance
        rho: sparsity
        x_val: value of non-zero entries

    Returns:
        MMSE estimate
    """
    # Log likelihood ratio
    log_ratio = (2 * x_val * r - x_val**2) / (2 * sigma_sq)
    log_prior_ratio = np.log(rho / (1 - rho + 1e-10) + 1e-10)
    log_odds = log_prior_ratio + log_ratio

    prob_nonzero = 1 / (1 + np.exp(-np.clip(log_odds, -500, 500)))

    return prob_nonzero * x_val


# =============================================================================
# AMP Algorithm
# =============================================================================

def amp_reconstruction(A, y, denoiser='soft', max_iter=100, tol=1e-6,
                       lam=None, rho=None, sigma_x=1.0, verbose=False):
    """
    Approximate Message Passing for compressed sensing.

    Solves: find x such that y ≈ Ax, with x sparse

    AMP iterations:
        z^t = y - A @ x^t + (1/delta) * z^{t-1} * <η'(r^{t-1})>
        r^t = A.T @ z^t + x^t
        x^{t+1} = η(r^t, τ^t)

    where:
        - δ = M/N (measurement ratio)
        - η is the denoising function
        - <η'> is the average derivative (Onsager correction)
        - τ^t is estimated from state evolution

    Args:
        A: measurement matrix (M x N), should have columns with norm sqrt(M)
        y: measurements (M,)
        denoiser: 'soft' for L1, 'gauss_bernoulli' for Bayesian optimal
        max_iter: maximum iterations
        tol: convergence tolerance
        lam: regularization parameter for soft thresholding
        rho: sparsity level (for Bayesian denoiser)
        sigma_x: std of non-zero entries (for Bayesian denoiser)
        verbose: print progress

    Returns:
        x_hat: reconstructed signal (N,)
        history: dict with convergence info
    """
    M, N = A.shape
    delta = M / N  # measurement ratio

    # For AMP with matrix entries ~ N(0, 1/N):
    # A.T @ z has variance ~ M/N * var(z) = delta * var(z)
    # We use A directly (already properly scaled)

    # Initialize
    x = np.zeros(N)
    z = y.copy()
    z_old = np.zeros(M)

    # For tracking
    history = {
        'mse': [],
        'residual': [],
        'x_estimates': []
    }

    # Estimate noise variance from measurements (if not provided)
    sigma_w_sq = 0.0  # assume noiseless for now

    for t in range(max_iter):
        # Effective observation: r = A^T z + x
        r = A.T @ z + x

        # Clip r to prevent overflow
        r = np.clip(r, -1e6, 1e6)

        # Estimate effective noise variance (state evolution)
        # sigma^2_t = sigma_w^2 + (1/delta) * MSE(x^t)
        # Use empirical estimate from residual
        sigma_sq = max(np.mean(z**2), 1e-10)
        sigma_sq = min(sigma_sq, 1e6)  # prevent overflow

        # Apply denoiser
        if denoiser == 'soft':
            # Threshold selection
            # Use a more conservative threshold that adapts during iterations
            if lam is None:
                # Start with moderate threshold, let state evolution guide it
                tau = np.sqrt(sigma_sq)  # simpler threshold
            else:
                tau = lam * np.sqrt(sigma_sq)

            x_new = soft_threshold(r, tau)
            onsager = np.mean(soft_threshold_derivative(r, tau))

        elif denoiser == 'gauss_bernoulli':
            if rho is None:
                raise ValueError("Must specify rho for Gauss-Bernoulli denoiser")

            x_new = gauss_bernoulli_denoiser(r, sigma_sq, rho, sigma_x)
            onsager = np.mean(gauss_bernoulli_denoiser_derivative(r, sigma_sq, rho, sigma_x))

        elif denoiser == 'bernoulli':
            if rho is None:
                raise ValueError("Must specify rho for Bernoulli denoiser")

            x_new = bernoulli_denoiser(r, sigma_sq, rho)
            onsager = np.mean((bernoulli_denoiser(r + 1e-6, sigma_sq, rho) -
                              bernoulli_denoiser(r - 1e-6, sigma_sq, rho)) / 2e-6)
        else:
            raise ValueError(f"Unknown denoiser: {denoiser}")

        # Clip onsager term to prevent instability
        onsager = np.clip(onsager, 0, 1)

        # Residual with Onsager correction
        # z = y - A @ x + (1/delta) * z_old * <eta'>
        z_new = y - A @ x_new + (z / delta) * onsager

        # Clip z to prevent overflow
        z_new = np.clip(z_new, -1e6, 1e6)

        # Check convergence
        diff = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-10)
        residual = np.linalg.norm(y - A @ x_new)

        history['residual'].append(residual)
        history['x_estimates'].append(x_new.copy())

        if verbose and t % 10 == 0:
            print(f"Iter {t}: residual = {residual:.6f}, diff = {diff:.6f}, "
                  f"nnz = {np.sum(np.abs(x_new) > 1e-6)}")

        # Check for divergence
        if residual > 1e10 or np.isnan(residual):
            if verbose:
                print(f"Warning: AMP diverged at iteration {t}")
            break

        if diff < tol:
            if verbose:
                print(f"Converged at iteration {t}")
            break

        x = x_new
        z_old = z
        z = z_new

    # Undo normalization effect on x
    x_hat = x_new

    return x_hat, history


# =============================================================================
# State Evolution (Theoretical Performance Prediction)
# =============================================================================

def state_evolution_soft_threshold(delta, rho, sigma_w=0, max_iter=100, tol=1e-8):
    """
    State evolution for AMP with soft thresholding.

    Predicts the MSE of AMP as a function of iteration.

    Args:
        delta: measurement ratio M/N
        rho: signal sparsity
        sigma_w: measurement noise std
        max_iter: max iterations
        tol: convergence tolerance

    Returns:
        mse_history: MSE at each iteration
        converged_mse: final MSE
    """
    # Initial MSE (assuming unit variance signal)
    mse = rho  # E[x^2] for Bernoulli-Gaussian with unit variance
    mse_history = [mse]

    for t in range(max_iter):
        # Effective noise variance
        sigma_sq = sigma_w**2 + mse / delta

        # Optimal threshold
        tau = np.sqrt(2 * np.log(1/rho)) * np.sqrt(sigma_sq)

        # MSE after denoising (approximation for soft thresholding)
        # This is simplified; exact formula involves integrals
        mse_new = rho * sigma_sq * np.exp(-tau**2 / (2 * sigma_sq))

        mse_history.append(mse_new)

        if abs(mse_new - mse) < tol:
            break

        mse = mse_new

    return mse_history, mse


# =============================================================================
# Measurement Matrix Construction
# =============================================================================

def generate_gaussian_matrix(M, N):
    """
    Generate Gaussian measurement matrix for AMP.

    Standard AMP formulation: entries iid N(0, 1/N)
    This ensures proper scaling for state evolution.

    Args:
        M: number of measurements
        N: signal dimension

    Returns:
        A: (M, N) measurement matrix
    """
    A = np.random.randn(M, N) / np.sqrt(N)
    return A


def generate_seeded_matrix(M, N, L, W, seed_fraction=0.1):
    """
    Generate spatially coupled (seeded) measurement matrix.

    This is key to achieving the information-theoretic limit
    as shown in Krzakala et al. (2012).

    The matrix has a band structure that enables "nucleation"
    of the correct solution from a seed region.

    Args:
        M: number of measurements
        N: signal dimension
        L: number of blocks
        W: coupling width
        seed_fraction: fraction of measurements in seed region

    Returns:
        A: spatially coupled measurement matrix
    """
    block_M = M // L
    block_N = N // L

    A = np.zeros((M, N))

    for l in range(L):
        for w in range(-W, W + 1):
            col_block = (l + w) % L
            row_start = l * block_M
            row_end = (l + 1) * block_M
            col_start = col_block * block_N
            col_end = (col_block + 1) * block_N

            # Coupling strength decreases with distance
            strength = 1.0 / (1 + abs(w))
            A[row_start:row_end, col_start:col_end] = \
                np.random.randn(block_M, block_N) * strength

    # Normalize
    A = A / np.sqrt(M)

    return A


# =============================================================================
# Demo and Testing
# =============================================================================

def demo_amp():
    """
    Demonstrate AMP reconstruction on a sparse signal.
    """
    np.random.seed(42)

    # Problem parameters
    N = 1000          # signal dimension
    rho = 0.1         # sparsity (10% non-zero)
    delta = 0.5       # measurement ratio M/N
    M = int(delta * N)
    sigma_w = 0.01    # measurement noise std

    print("=" * 60)
    print("AMP Compressed Sensing Demo")
    print("=" * 60)
    print(f"Signal dimension N = {N}")
    print(f"Sparsity rho = {rho}")
    print(f"Measurements M = {M} (delta = {delta})")
    print(f"Noise level sigma = {sigma_w}")
    print()

    # Generate sparse signal (Gauss-Bernoulli)
    support = np.random.rand(N) < rho
    x_true = np.zeros(N)
    x_true[support] = np.random.randn(np.sum(support))

    print(f"True signal: {np.sum(support)} non-zero entries")

    # Generate measurement matrix
    A = generate_gaussian_matrix(M, N)

    # Measurements
    y = A @ x_true + sigma_w * np.random.randn(M)

    # Reconstruct with soft thresholding
    print("\n--- AMP with Soft Thresholding ---")
    x_soft, hist_soft = amp_reconstruction(
        A, y, denoiser='soft', max_iter=200, verbose=True, lam=0.5
    )
    mse_soft = np.mean((x_soft - x_true)**2)
    print(f"Final MSE (soft): {mse_soft:.6f}")

    # Reconstruct with Bayesian optimal denoiser
    print("\n--- AMP with Gauss-Bernoulli Denoiser ---")
    x_bayes, hist_bayes = amp_reconstruction(
        A, y, denoiser='gauss_bernoulli', max_iter=200,
        rho=rho, sigma_x=1.0, verbose=True
    )
    mse_bayes = np.mean((x_bayes - x_true)**2)
    print(f"Final MSE (Bayesian): {mse_bayes:.6f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # True vs reconstructed signal
    ax = axes[0, 0]
    ax.stem(x_true[:100], linefmt='b-', markerfmt='bo', basefmt='k-', label='True')
    ax.stem(x_soft[:100], linefmt='r-', markerfmt='rx', basefmt='k-', label='Soft')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Signal Reconstruction (Soft Thresholding)')
    ax.legend()

    ax = axes[0, 1]
    ax.stem(x_true[:100], linefmt='b-', markerfmt='bo', basefmt='k-', label='True')
    ax.stem(x_bayes[:100], linefmt='g-', markerfmt='g^', basefmt='k-', label='Bayesian')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Signal Reconstruction (Bayesian Optimal)')
    ax.legend()

    # Convergence
    ax = axes[1, 0]
    ax.semilogy(hist_soft['residual'], 'r-', label='Soft')
    ax.semilogy(hist_bayes['residual'], 'g-', label='Bayesian')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual ||y - Ax||')
    ax.set_title('Convergence')
    ax.legend()
    ax.grid(True)

    # Scatter plot
    ax = axes[1, 1]
    ax.scatter(x_true, x_soft, alpha=0.5, s=10, label=f'Soft (MSE={mse_soft:.4f})')
    ax.scatter(x_true, x_bayes, alpha=0.5, s=10, label=f'Bayesian (MSE={mse_bayes:.4f})')
    ax.plot([-3, 3], [-3, 3], 'k--', label='Perfect')
    ax.set_xlabel('True x')
    ax.set_ylabel('Reconstructed x')
    ax.set_title('Reconstruction Quality')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('amp_demo.png', dpi=150)
    plt.show()

    print(f"\nPlot saved to 'amp_demo.png'")

    return x_true, x_soft, x_bayes


def phase_transition_experiment():
    """
    Generate phase transition diagram showing success/failure regions.

    This reproduces a key result from Krzakala et al. (2012):
    the phase diagram in (delta, rho) space.
    """
    print("\n" + "=" * 60)
    print("Phase Transition Experiment")
    print("=" * 60)

    N = 200  # Smaller for faster computation
    n_trials = 3

    # Grid of (delta, rho) values
    deltas = np.linspace(0.2, 0.8, 7)
    rhos = np.linspace(0.05, 0.35, 7)

    success_rate = np.zeros((len(rhos), len(deltas)))

    for i, rho in enumerate(rhos):
        for j, delta in enumerate(deltas):
            M = int(delta * N)
            successes = 0

            for trial in range(n_trials):
                # Generate problem
                support = np.random.rand(N) < rho
                x_true = np.zeros(N)
                x_true[support] = np.random.randn(np.sum(support))

                A = generate_gaussian_matrix(M, N)
                y = A @ x_true

                # Reconstruct
                x_hat, _ = amp_reconstruction(
                    A, y, denoiser='gauss_bernoulli',
                    rho=rho, max_iter=100, verbose=False
                )

                # Check success (normalized MSE < threshold)
                nmse = np.mean((x_hat - x_true)**2) / np.mean(x_true**2)
                if nmse < 0.01:
                    successes += 1

            success_rate[i, j] = successes / n_trials
            print(f"rho={rho:.2f}, delta={delta:.2f}: {success_rate[i,j]*100:.0f}% success")

    # Plot phase diagram
    plt.figure(figsize=(8, 6))
    plt.imshow(success_rate, extent=[deltas[0], deltas[-1], rhos[-1], rhos[0]],
               aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='Success Rate')
    plt.xlabel('Measurement ratio δ = M/N')
    plt.ylabel('Sparsity ρ')
    plt.title('AMP Phase Transition Diagram\n(Green = Success, Red = Failure)')

    # Add theoretical Donoho-Tanner limit (approximate)
    delta_theory = np.linspace(0.1, 0.9, 100)
    rho_theory = delta_theory * 0.5  # Simplified; actual limit is more complex
    plt.plot(delta_theory, rho_theory, 'b--', linewidth=2, label='Approx. DT limit')
    plt.legend()

    plt.savefig('phase_transition.png', dpi=150)
    plt.show()

    print(f"\nPhase diagram saved to 'phase_transition.png'")


if __name__ == "__main__":
    # Run demo
    x_true, x_soft, x_bayes = demo_amp()

    # Uncomment to run phase transition experiment (takes longer)
    # phase_transition_experiment()
