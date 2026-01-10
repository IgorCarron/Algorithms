"""
Implementation of: Simpler and Better Cardinality Estimators for HyperLogLog and PCSA

ArXiv: https://arxiv.org/abs/2208.10578v1

Authors:
    - Seth Pettie
    - Dingyu Wang

Abstract:
    Cardinality Estimation (aka Distinct Elements) is a classic problem in sketching with many industrial applications.
    Although sketching algorithms are fairly simple, analyzing the cardinality estimators is notoriously difficult,
    and even today the state-of-the-art sketches such as HyperLogLog and (compressed) PCSA are not covered in
    graduate level Big Data courses. In this paper we define a class of generalized remaining area (τ-GRA) estimators,
    and observe that HyperLogLog, LogLog, and some estimators for PCSA are merely instantiations of τ-GRA for
    various integral values of τ. We then analyze the limiting relative variance of τ-GRA estimators.

This implementation provides the τ-GRA framework for cardinality estimation with HyperLogLog and PCSA sketches,
including variance analysis and optimal τ parameter selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.special import gamma
from scipy.optimize import minimize_scalar
import hashlib
from collections import defaultdict


class TauGRAEstimator:
    """
    τ-GRA (Generalized Remaining Area) estimator for cardinality estimation.
    
    This class implements the τ-GRA framework that unifies HyperLogLog and PCSA
    estimators and provides improved variance through optimal τ selection.
    """
    
    def __init__(self, m: int, tau: float, sketch_type: str = 'hyperloglog'):
        """
        Initialize the τ-GRA estimator.
        
        Args:
            m: Number of subsketches/buckets
            tau: The τ parameter for generalized remaining area
            sketch_type: Either 'hyperloglog' or 'pcsa'
        """
        self.m = m
        self.tau = tau
        self.sketch_type = sketch_type.lower()
        
        if self.sketch_type == 'hyperloglog':
            # Random offsets for HyperLogLog
            self.offsets = np.random.uniform(0, 1, m)
            self.sketch_state = np.zeros(m)  # Maximum leading zeros in each bucket
        elif self.sketch_type == 'pcsa':
            # Uniform offsets for PCSA
            self.offsets = np.arange(m) / m
            self.sketch_state = [defaultdict(bool) for _ in range(m)]  # Bit vectors
        else:
            raise ValueError("sketch_type must be 'hyperloglog' or 'pcsa'")
    
    def _hash_element(self, element: str) -> Tuple[int, float]:
        """
        Hash an element and extract bucket index and geometric random variable.
        
        Args:
            element: String element to hash
            
        Returns:
            Tuple of (bucket_index, geometric_value)
        """
        # Use SHA-256 for high quality hashing
        hash_bytes = hashlib.sha256(element.encode()).digest()
        
        # Extract bucket index from first log2(m) bits
        bucket_bits = int(np.log2(self.m))
        bucket_index = int.from_bytes(hash_bytes[:4], 'big') % self.m
        
        # Extract geometric random variable from remaining bits
        # Count leading zeros to simulate geometric distribution
        remaining_bytes = hash_bytes[4:]
        leading_zeros = 0
        for byte in remaining_bytes:
            for bit in range(8):
                if (byte >> (7 - bit)) & 1 == 0:
                    leading_zeros += 1
                else:
                    return bucket_index, leading_zeros
        
        return bucket_index, leading_zeros
    
    def update(self, element: str) -> None:
        """
        Update the sketch with a new element.
        
        Args:
            element: String element to add to the sketch
        """
        bucket_idx, geom_value = self._hash_element(element)
        
        if self.sketch_type == 'hyperloglog':
            # Update maximum leading zeros for this bucket
            self.sketch_state[bucket_idx] = max(self.sketch_state[bucket_idx], geom_value)
        
        elif self.sketch_type == 'pcsa':
            # Set bit at position geom_value in bucket bucket_idx
            self.sketch_state[bucket_idx][geom_value] = True
    
    def _compute_tau_gra(self) -> float:
        """
        Compute the τ-generalized remaining area of the current sketch.
        
        Returns:
            The τ-GRA value
        """
        if self.sketch_type == 'hyperloglog':
            # For HyperLogLog: τ-GRA = sum of (2^(-τ(R_i + X_i)))
            tau_gra = 0.0
            for i in range(self.m):
                tau_gra += 2**(-self.tau * (self.offsets[i] + self.sketch_state[i]))
            return tau_gra
        
        elif self.sketch_type == 'pcsa':
            # For PCSA: τ-GRA = sum over free cells of cell_size^τ
            tau_gra = 0.0
            for i in range(self.m):
                # Simulate infinite dartboard by checking reasonable range
                for j in range(-20, 50):  # Practical range for cell indices
                    if not self.sketch_state[i][j]:  # Cell is free
                        cell_size = 2**(-j - self.offsets[i])
                        tau_gra += cell_size**self.tau
            return tau_gra
    
    def estimate_cardinality(self) -> float:
        """
        Estimate the cardinality using τ-GRA.
        
        Returns:
            Estimated cardinality
        """
        tau_gra_sum = self._compute_tau_gra()
        
        if self.sketch_type == 'hyperloglog':
            # Normalization factor from Theorem 3
            normalization = self.m * (gamma(self.tau) / (1 - 2**(-self.tau)) / np.log(2))**(1/self.tau)
            return normalization * (tau_gra_sum / self.m)**(-1/self.tau)
        
        elif self.sketch_type == 'pcsa':
            # Normalization factor from Theorem 4
            normalization = self.m * (gamma(self.tau) / np.log(2))**(1/self.tau)
            return normalization * (tau_gra_sum / self.m)**(-1/self.tau)


def compute_relative_variance_hyperloglog(tau: float) -> float:
    """
    Compute the limiting relative variance for HyperLogLog with given τ.
    
    Args:
        tau: The τ parameter
        
    Returns:
        Relative variance coefficient
    """
    if tau <= 0:
        return float('inf')
    
    numerator = gamma(2*tau) / np.log(2) / gamma(tau)**2
    denominator = (1 + 2**(-tau)) / (1 - 2**(-tau)) - 1
    return (1/tau**2) * numerator * denominator


def compute_relative_variance_pcsa(tau: float) -> float:
    """
    Compute the limiting relative variance for PCSA with given τ.
    
    Args:
        tau: The τ parameter
        
    Returns:
        Relative variance coefficient
    """
    if tau <= 0:
        return float('inf')
    
    return (1 - 2**(-2*tau)) * gamma(2*tau) / (np.log(2) * tau**2 * gamma(tau)**2)


def find_optimal_tau(sketch_type: str) -> Tuple[float, float]:
    """
    Find the optimal τ value that minimizes relative variance.
    
    Args:
        sketch_type: Either 'hyperloglog' or 'pcsa'
        
    Returns:
        Tuple of (optimal_tau, minimum_variance)
    """
    if sketch_type == 'hyperloglog':
        variance_func = compute_relative_variance_hyperloglog
    else:
        variance_func = compute_relative_variance_pcsa
    
    result = minimize_scalar(variance_func, bounds=(0.01, 2.0), method='bounded')
    return result.x, result.fun


def run_cardinality_estimation_experiment(true_cardinality: int, m: int = 64) -> None:
    """
    Run experiment comparing different τ values for cardinality estimation.
    
    Args:
        true_cardinality: The true number of distinct elements
        m: Number of subsketches
    """
    # Generate test data
    elements = [f"element_{i}" for i in range(true_cardinality)]
    
    # Test different tau values
    tau_values = [0.5, 1.0, 1.5]
    
    # Add optimal tau values
    optimal_tau_hll, _ = find_optimal_tau('hyperloglog')
    optimal_tau_pcsa, _ = find_optimal_tau('pcsa')
    
    tau_values.extend([optimal_tau_hll, optimal_tau_pcsa])
    
    results = []
    
    for tau in tau_values:
        # Test HyperLogLog
        estimator_hll = TauGRAEstimator(m, tau, 'hyperloglog')
        for element in elements:
            estimator_hll.update(element)
        estimate_hll = estimator_hll.estimate_cardinality()
        
        # Test PCSA
        estimator_pcsa = TauGRAEstimator(m, tau, 'pcsa')
        for element in elements:
            estimator_pcsa.update(element)
        estimate_pcsa = estimator_pcsa.estimate_cardinality()
        
        results.append({
            'tau': tau,
            'hll_estimate': estimate_hll,
            'pcsa_estimate': estimate_pcsa,
            'hll_error': abs(estimate_hll - true_cardinality) / true_cardinality,
            'pcsa_error': abs(estimate_pcsa - true_cardinality) / true_cardinality
        })
    
    return results


if __name__ == "__main__":
    # Generate variance plots similar to Figures 2 and 3 in the paper
    
    # Figure 1: Relative variance for HyperLogLog (similar to Figure 2)
    tau_range = np.linspace(0.1, 2.0, 200)
    hll_variances = [compute_relative_variance_hyperloglog(tau) for tau in tau_range]
    
    # Find optimal tau and Cramér-Rao bound
    optimal_tau_hll, min_var_hll = find_optimal_tau('hyperloglog')
    cramer_rao_hll = np.log(2) * (np.pi**2 / 6) - 1  # From paper
    
    plt.figure(figsize=(10, 6))
    plt.plot(tau_range, hll_variances, 'b-', linewidth=2, label='τ-GRA')
    plt.axhline(y=cramer_rao_hll, color='r', linestyle='--', linewidth=2, label='Cramér-Rao lower bound')
    plt.axhline(y=3*np.log(2) - 1, color='g', linestyle=':', linewidth=2, label='HyperLogLog (τ=1)')
    plt.axvline(x=optimal_tau_hll, color='orange', linestyle='-.', alpha=0.7, label=f'Optimal τ* = {optimal_tau_hll:.3f}')
    plt.scatter([optimal_tau_hll], [min_var_hll], color='orange', s=100, zorder=5)
    
    plt.xlabel('τ', fontsize=12)
    plt.ylabel('Relative variance', fontsize=12)
    plt.title('Relative variance of estimators for the LogLog sketch', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.1, 2.0)
    plt.ylim(1.0, 2.0)
    plt.show()
    
    # Figure 2: Relative variance for PCSA (similar to Figure 3)
    pcsa_variances = [compute_relative_variance_pcsa(tau) for tau in tau_range]
    
    # Find optimal tau and bounds for PCSA
    optimal_tau_pcsa, min_var_pcsa = find_optimal_tau('pcsa')
    cramer_rao_pcsa = (np.pi**2 / 6) / np.log(2)  # From paper
    
    plt.figure(figsize=(10, 6))
    plt.plot(tau_range, pcsa_variances, 'b-', linewidth=2, label='τ-GRA')
    plt.axhline(y=cramer_rao_pcsa, color='r', linestyle='--', linewidth=2, label='Cramér-Rao lower bound')
    plt.axhline(y=0.6, color='purple', linestyle=':', linewidth=2, label="Flajolet & Martin's First Zero")
    plt.axhline(y=np.log(2)**2, color='g', linestyle=':', linewidth=2, label="Lang's coupon collector")
    plt.axhline(y=3*(np.log(2)**2)/4, color='orange', linestyle=':', linewidth=2, label='Remaining area (τ=1)')
    plt.axvline(x=optimal_tau_pcsa, color='red', linestyle='-.', alpha=0.7, label=f'Optimal τ* = {optimal_tau_pcsa:.3f}')
    plt.scatter([optimal_tau_pcsa], [min_var_pcsa], color='red', s=100, zorder=5)
    
    plt.xlabel('τ', fontsize=12)
    plt.ylabel('Relative variance', fontsize=12)
    plt.title('Relative variance of estimators for the PCSA sketch', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.1, 2.0)
    plt.ylim(0.3, 1.0)
    plt.show()
    
    # Figure 3: Cardinality estimation accuracy comparison
    true_cardinalities = [100, 500, 1000, 5000, 10000]
    estimation_errors = {'HyperLogLog (τ=1)': [], 'HyperLogLog (optimal τ)': [], 
                        'PCSA (τ=0)': [], 'PCSA (optimal τ)': []}
    
    for true_card in true_cardinalities:
        # Generate test elements
        elements = [f"element_{i}_{np.random.randint(0, 1000000)}" for i in range(true_card)]
        
        # Test standard HyperLogLog (τ=1)
        hll_standard = TauGRAEstimator(64, 1.0, 'hyperloglog')
        for elem in elements:
            hll_standard.update(elem)
        error_hll_std = abs(hll_standard.estimate_cardinality() - true_card) / true_card
        estimation_errors['HyperLogLog (τ=1)'].append(error_hll_std)
        
        # Test optimal HyperLogLog
        hll_optimal = TauGRAEstimator(64, optimal_tau_hll, 'hyperloglog')
        for elem in elements:
            hll_optimal.update(elem)
        error_hll_opt = abs(hll_optimal.estimate_cardinality() - true_card) / true_card
        estimation_errors['HyperLogLog (optimal τ)'].append(error_hll_opt)
        
        # Test PCSA with τ≈0 (counting approach)
        pcsa_counting = TauGRAEstimator(64, 0.1, 'pcsa')  # τ=0.1 approximates τ→0
        for elem in elements:
            pcsa_counting.update(elem)
        error_pcsa_count = abs(pcsa_counting.estimate_cardinality() - true_card) / true_card
        estimation_errors['PCSA (τ=0)'].append(error_pcsa_count)
        
        # Test optimal PCSA
        pcsa_optimal = TauGRAEstimator(64, optimal_tau_pcsa, 'pcsa')
        for elem in elements:
            pcsa_optimal.update(elem)
        error_pcsa_opt = abs(pcsa_optimal.estimate_cardinality() - true_card) / true_card
        estimation_errors['PCSA (optimal τ)'].append(error_pcsa_opt)
    
    plt.figure(figsize=(12, 6))
    for method, errors in estimation_errors.items():
        plt.plot(true_cardinalities, errors, 'o-', linewidth=2, markersize=6, label=method)
    
    plt.xlabel('True Cardinality', fontsize=12)
    plt.ylabel('Relative Error', fontsize=12)
    plt.title('Cardinality Estimation Accuracy Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    # Print summary results
    print("\n=== τ-GRA Cardinality Estimation Results ===")
    print(f"\nOptimal τ for HyperLogLog: {optimal_tau_hll:.6f}")
    print(f"Minimum relative variance (HyperLogLog): {min_var_hll:.6f}")
    print(f"Standard HyperLogLog variance (τ=1): {compute_relative_variance_hyperloglog(1.0):.6f}")
    print(f"Cramér-Rao bound (HyperLogLog): {cramer_rao_hll:.6f}")
    
    print(f"\nOptimal τ for PCSA: {optimal_tau_pcsa:.6f}")
    print(f"Minimum relative variance (PCSA): {min_var_pcsa:.6f}")
    print(f"Lang's coupon collector variance (τ→0): {np.log(2)**2:.6f}")
    print(f"Cramér-Rao bound (PCSA): {cramer_rao_pcsa:.6f}")
    
    improvement_hll = (compute_relative_variance_hyperloglog(1.0) - min_var_hll) / compute_relative_variance_hyperloglog(1.0) * 100
    improvement_pcsa = (np.log(2)**2 - min_var_pcsa) / np.log(2)**2 * 100
    
    print(f"\nImprovement over standard HyperLogLog: {improvement_hll:.2f}%")
    print(f"Improvement over Lang's estimator: {improvement_pcsa:.2f}%")
