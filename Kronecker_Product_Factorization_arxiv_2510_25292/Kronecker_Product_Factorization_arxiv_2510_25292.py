"""
Implementation of: Identifying Kronecker product factorizations

ArXiv: https://arxiv.org/abs/2510.25292

Authors:
    - Yannis Voet
    - Leonardo De Novellis

Abstract:
    The Kronecker product is an invaluable tool for data-sparse representations of large networks and matrices with countless applications in machine learning, graph theory and numerical linear algebra. In some instances, the sparsity pattern of large matrices may already hide a Kronecker product. Similarly, a large network, represented by its adjacency matrix, may sometimes be factorized as a Kronecker product of smaller adjacency matrices. In this article, we determine all possible Kronecker facto...

This implementation provides algorithms for finding all possible Kronecker product factorizations
of binary matrices and visualizing them through decomposition graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Set, Dict, Optional
from itertools import product
from collections import defaultdict


class KroneckerFactorizer:
    """
    A class for identifying Kronecker product factorizations of binary matrices.
    
    This implementation follows the algorithm described in the paper for finding
    all possible ways to decompose a binary matrix A as a Kronecker product
    A = A1 ⊗ A2 ⊗ ... ⊗ Aℓ.
    """
    
    def __init__(self, matrix: np.ndarray):
        """
        Initialize the factorizer with a binary matrix.
        
        Args:
            matrix: Binary matrix (0s and 1s) to factorize
        """
        self.matrix = matrix.astype(int)
        self.n = matrix.shape[0]
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")
        
        # Get sparsity pattern
        self.sparsity_pattern = self._get_sparsity_pattern()
        
        # Find all divisors
        self.divisors = self._get_divisors(self.n)
        
        # Compatible pairs (n1, n2) where n1 * n2 = n
        self.compatible_pairs = self._get_compatible_pairs()
        
        # Store factorizations
        self.length_2_factorizations = []
        self.prime_factorizations = []
        
    def _get_sparsity_pattern(self) -> List[Tuple[int, int]]:
        """Extract sparsity pattern from matrix."""
        rows, cols = np.where(self.matrix == 1)
        return list(zip(rows + 1, cols + 1))  # 1-indexed
    
    def _get_divisors(self, n: int) -> List[int]:
        """Get all divisors of n."""
        divisors = []
        for i in range(1, int(np.sqrt(n)) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
        return sorted(divisors)
    
    def _get_compatible_pairs(self) -> List[Tuple[int, int]]:
        """Get all compatible pairs (n1, n2) where n1 * n2 = n and n1, n2 > 1."""
        pairs = []
        for n1 in self.divisors:
            if n1 > 1 and n1 < self.n:
                n2 = self.n // n1
                if n2 > 1:
                    pairs.append((n1, n2))
        return pairs
    
    def _linear_index(self, i: int, j: int, n: int) -> int:
        """Convert (i,j) matrix indices to linear index."""
        return (j - 1) * n + i
    
    def _index_pair_from_linear(self, linear_idx: int, n: int) -> Tuple[int, int]:
        """Convert linear index back to (i,j) matrix indices."""
        i = ((linear_idx - 1) % n) + 1
        j = ((linear_idx - 1) // n) + 1
        return i, j
    
    def _check_factorizability(self, n1: int, n2: int) -> Tuple[bool, Optional[Tuple[Set[int], Set[int]]]]:
        """
        Check if matrix admits an (n1, n2) factorization using Lemma 3.5.
        
        Returns:
            Tuple of (is_factorizable, (S1, S2)) where S1 x S2 = S if factorizable
        """
        S = set()
        
        # For each nonzero entry, compute corresponding indices
        for i, j in self.sparsity_pattern:
            # Euclidean division to get factor indices
            i1 = ((i - 1) // n2) + 1
            j1 = ((j - 1) // n2) + 1
            i2 = ((i - 1) % n2) + 1
            j2 = ((j - 1) % n2) + 1
            
            # Convert to linear indices
            l1 = self._linear_index(i1, j1, n1)
            l2 = self._linear_index(i2, j2, n2)
            
            S.add((l1, l2))
        
        # Try to write S as Cartesian product S1 x S2
        if not S:
            return True, (set(), set())
        
        # Extract potential S1 and S2
        S1_candidate = set(pair[0] for pair in S)
        S2_candidate = set(pair[1] for pair in S)
        
        # Check if S = S1 x S2
        cartesian_product = set(product(S1_candidate, S2_candidate))
        
        if S == cartesian_product:
            return True, (S1_candidate, S2_candidate)
        else:
            return False, None
    
    def find_all_length_2_factorizations(self) -> List[Tuple[int, int]]:
        """
        Find all length-2 factorizations of the matrix.
        
        Returns:
            List of (n1, n2) pairs for which the matrix is factorizable
        """
        factorizations = []
        
        for n1, n2 in self.compatible_pairs:
            is_factorizable, _ = self._check_factorizability(n1, n2)
            if is_factorizable:
                factorizations.append((n1, n2))
        
        self.length_2_factorizations = factorizations
        return factorizations
    
    def _build_factorization_branches(self) -> List[List[Tuple[int, int]]]:
        """
        Build factorization branches using Corollary 3.11.
        
        Returns:
            List of branches, where each branch is a list of (l, r) pairs
        """
        if not self.length_2_factorizations:
            return []
        
        # Extract left and right indices
        L = [pair[0] for pair in self.length_2_factorizations]
        R = [pair[1] for pair in self.length_2_factorizations]
        
        # Create mapping from left index to right index
        lr_map = dict(self.length_2_factorizations)
        
        branches = []
        used_pairs = set()
        
        # Build branches by following multiples
        for l0 in sorted(set(L)):
            if (l0, lr_map[l0]) in used_pairs:
                continue
            
            branch = [(l0, lr_map[l0])]
            used_pairs.add((l0, lr_map[l0]))
            
            current_l = l0
            while True:
                # Look for multiples of current_l in L
                found_multiple = False
                for l_next in sorted(set(L)):
                    if l_next > current_l and l_next % current_l == 0 and (l_next, lr_map[l_next]) not in used_pairs:
                        # Check if this forms a valid continuation
                        if l_next * lr_map[l_next] == self.n:
                            branch.append((l_next, lr_map[l_next]))
                            used_pairs.add((l_next, lr_map[l_next]))
                            current_l = l_next
                            found_multiple = True
                            break
                
                if not found_multiple:
                    break
            
            branches.append(branch)
        
        return branches
    
    def find_prime_factorizations(self) -> List[Tuple[int, ...]]:
        """
        Find all prime factorizations by combining length-2 factorizations.
        
        Returns:
            List of prime factorizations as tuples of factor sizes
        """
        # First find all length-2 factorizations
        self.find_all_length_2_factorizations()
        
        if not self.length_2_factorizations:
            return []
        
        # Build branches
        branches = self._build_factorization_branches()
        
        prime_factorizations = []
        
        for branch in branches:
            if len(branch) == 1:
                # Single length-2 factorization
                prime_factorizations.append(branch[0])
            else:
                # Combine factorizations in branch
                # Extract the pattern: (l, p1*p2*...*pq*r), (p1*l, p2*...*pq*r), ..., (p1*p2*...*pq*l, r)
                factors = [branch[0][0]]  # Start with l
                
                for i in range(len(branch) - 1):
                    # Extract pi from the ratio
                    pi = branch[i+1][0] // branch[i][0]
                    factors.append(pi)
                
                # Add the final r
                factors.append(branch[-1][1])
                
                prime_factorizations.append(tuple(factors))
        
        self.prime_factorizations = prime_factorizations
        return prime_factorizations
    
    def is_maximal(self) -> bool:
        """
        Check if the matrix is maximal (admits factorization for all compatible pairs).
        
        Returns:
            True if matrix is maximal, False otherwise
        """
        self.find_all_length_2_factorizations()
        return len(self.length_2_factorizations) == len(self.compatible_pairs)
    
    def create_decomposition_graph(self) -> nx.DiGraph:
        """
        Create the decomposition graph as described in Section 4.
        
        Returns:
            NetworkX directed graph representing the decomposition structure
        """
        self.find_all_length_2_factorizations()
        branches = self._build_factorization_branches()
        
        G = nx.DiGraph()
        
        # Add all left indices as vertices
        L = list(set(pair[0] for pair in self.length_2_factorizations))
        G.add_nodes_from(L)
        
        # Add edges for each branch
        edge_colors = []
        edge_labels = {}
        
        for branch_idx, branch in enumerate(branches):
            color = plt.cm.tab10(branch_idx % 10)
            
            for i in range(len(branch) - 1):
                l_current = branch[i][0]
                l_next = branch[i+1][0]
                weight = l_next // l_current
                
                G.add_edge(l_current, l_next, weight=weight, branch=branch_idx)
                edge_labels[(l_current, l_next)] = str(weight)
                edge_colors.append(color)
        
        return G


def create_basis_matrix(i: int, j: int, n: int) -> np.ndarray:
    """
    Create basis matrix E_ij of size n x n.
    
    Args:
        i, j: Position of the single 1 (1-indexed)
        n: Size of matrix
        
    Returns:
        Binary matrix with single 1 at position (i,j)
    """
    matrix = np.zeros((n, n), dtype=int)
    matrix[i-1, j-1] = 1
    return matrix


def visualize_decomposition_graph(G: nx.DiGraph, title: str = "Decomposition Graph"):
    """
    Visualize the decomposition graph.
    
    Args:
        G: NetworkX directed graph
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create layout
    if len(G.nodes()) <= 6:
        pos = nx.spring_layout(G, k=2, iterations=50)
    else:
        pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.7)
    
    # Draw edges with different colors for different branches
    edge_colors = []
    for edge in G.edges():
        branch_id = G.edges[edge].get('branch', 0)
        edge_colors.append(plt.cm.tab10(branch_id % 10))
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                          arrows=True, arrowsize=20, arrowstyle='->')
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edge labels (weights)
    edge_labels = {}
    for edge in G.edges():
        weight = G.edges[edge].get('weight', '')
        edge_labels[edge] = str(weight)
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def demonstrate_algorithm():
    """
    Demonstrate the Kronecker factorization algorithm with examples from the paper.
    """
    print("=" * 60)
    print("Kronecker Product Factorization Algorithm Demonstration")
    print("=" * 60)
    
    # Example 1: Matrix from Example 2.5 in the paper
    print("\nExample 1: Matrix with multiple prime factorizations")
    print("-" * 50)
    
    # Create a matrix that admits (2,2,3), (2,3,2), and (3,2,2) factorizations
    # This is a maximal matrix of size 12
    A1 = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    B1 = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    
    # Create Kronecker product
    matrix1 = np.kron(A1, B1)
    print(f"Matrix size: {matrix1.shape}")
    print(f"Sparsity: {np.sum(matrix1)} / {matrix1.size} = {np.sum(matrix1)/matrix1.size:.3f}")
    
    factorizer1 = KroneckerFactorizer(matrix1)
    
    # Find factorizations
    length2_facts = factorizer1.find_all_length_2_factorizations()
    prime_facts = factorizer1.find_prime_factorizations()
    is_maximal = factorizer1.is_maximal()
    
    print(f"Length-2 factorizations: {length2_facts}")
    print(f"Prime factorizations: {prime_facts}")
    print(f"Is maximal: {is_maximal}")
    
    # Visualize decomposition graph
    G1 = factorizer1.create_decomposition_graph()
    visualize_decomposition_graph(G1, "Example 1: Multiple Prime Factorizations")
    
    # Example 2: Simple basis matrix (Example from paper)
    print("\nExample 2: Basis matrix E_11 (4x4)")
    print("-" * 40)
    
    matrix2 = create_basis_matrix(1, 1, 4)
    print(f"Matrix:")
    print(matrix2)
    
    factorizer2 = KroneckerFactorizer(matrix2)
    length2_facts2 = factorizer2.find_all_length_2_factorizations()
    prime_facts2 = factorizer2.find_prime_factorizations()
    
    print(f"Length-2 factorizations: {length2_facts2}")
    print(f"Prime factorizations: {prime_facts2}")
    
    if length2_facts2:
        G2 = factorizer2.create_decomposition_graph()
        visualize_decomposition_graph(G2, "Example 2: Basis Matrix E_11")
    
    # Example 3: Non-factorizable matrix
    print("\nExample 3: Non-factorizable matrix")
    print("-" * 35)
    
    # Create a matrix that cannot be factorized (from Example 3.6)
    matrix3 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ])
    
    print(f"Matrix:")
    print(matrix3)
    
    factorizer3 = KroneckerFactorizer(matrix3)
    length2_facts3 = factorizer3.find_all_length_2_factorizations()
    prime_facts3 = factorizer3.find_prime_factorizations()
    
    print(f"Length-2 factorizations: {length2_facts3}")
    print(f"Prime factorizations: {prime_facts3}")
    
    if not prime_facts3:
        print("Matrix is prime (cannot be factorized)")
    
    # Visualization of sparsity patterns
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(matrix1, cmap='Blues', interpolation='nearest')
    plt.title('Example 1: Maximal Matrix\n(Multiple Factorizations)')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(matrix2, cmap='Blues', interpolation='nearest')
    plt.title('Example 2: Basis Matrix E_11')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(matrix3, cmap='Blues', interpolation='nearest')
    plt.title('Example 3: Prime Matrix\n(Non-factorizable)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.suptitle('Sparsity Patterns of Test Matrices', y=1.02, fontsize=16)
    plt.show()
    
    # Statistics comparison
    matrices = [matrix1, matrix2, matrix3]
    factorizers = [factorizer1, factorizer2, factorizer3]
    names = ['Maximal Matrix', 'Basis Matrix', 'Prime Matrix']
    
    print("\nComparison Summary:")
    print("=" * 50)
    print(f"{'Matrix':<15} {'Size':<6} {'Sparsity':<10} {'# Length-2':<12} {'# Prime':<8} {'Maximal':<8}")
    print("-" * 65)
    
    for i, (matrix, factorizer, name) in enumerate(zip(matrices, factorizers, names)):
        size = matrix.shape[0]
        sparsity = f"{np.sum(matrix)}/{matrix.size}"
        num_length2 = len(factorizer.length_2_factorizations)
        num_prime = len(factorizer.prime_factorizations)
        maximal = "Yes" if factorizer.is_maximal() else "No"
        
        print(f"{name:<15} {size:<6} {sparsity:<10} {num_length2:<12} {num_prime:<8} {maximal:<8}")


if __name__ == "__main__":
    demonstrate_algorithm()