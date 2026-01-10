"""
Implementation of: Identifying Kronecker product factorizations

ArXiv: https://arxiv.org/abs/2510.25292

Authors:
    - Yannis Voet
    - Leonardo De Novellis

Abstract:
    The Kronecker product is an invaluable tool for data-sparse representations of large networks and matrices with countless applications in machine learning, graph theory and numerical linear algebra. In some instances, the sparsity pattern of large matrices may already hide a Kronecker product. Similarly, a large network, represented by its adjacency matrix, may sometimes be factorized as a Kronecker product of smaller adjacency matrices. In this article, we determine all possible Kronecker factorizations of a binary matrix and visualize them through its decomposition graph. Such sparsity-informed factorizations may later enable good (approximate) Kronecker factorizations of real matrices or reveal the latent structure of a network.

This implementation provides algorithms to find all length-2 Kronecker factorizations of binary matrices,
combine them to find longer factorizations, and visualize the decomposition structure through graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Set, Dict, Optional
from itertools import product
import math

def get_divisors(n: int) -> List[int]:
    """Get all divisors of n."""
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)

def get_compatible_pairs(n: int) -> List[Tuple[int, int]]:
    """Get all compatible pairs (n1, n2) such that n1 * n2 = n and n1, n2 > 1."""
    divisors = get_divisors(n)
    pairs = []
    for n1 in divisors:
        if n1 > 1 and n1 < n:
            n2 = n // n1
            if n2 > 1:
                pairs.append((n1, n2))
    return pairs

def linear_index(i: int, j: int, n: int) -> int:
    """Convert (i,j) matrix indices to linear index."""
    return (j - 1) * n + i

def index_to_pair(idx: int, n: int) -> Tuple[int, int]:
    """Convert linear index to (i,j) matrix indices."""
    i = ((idx - 1) % n) + 1
    j = ((idx - 1) // n) + 1
    return i, j

def decompose_indices(i: int, j: int, n2: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Decompose matrix indices for Kronecker factorization check."""
    # Convert to 0-based indexing
    i_zero = i - 1
    j_zero = j - 1
    
    # Euclidean division
    i1_tilde = i_zero // n2
    i2_tilde = i_zero % n2
    j1_tilde = j_zero // n2
    j2_tilde = j_zero % n2
    
    # Convert back to 1-based indexing
    return (i1_tilde + 1, j1_tilde + 1), (i2_tilde + 1, j2_tilde + 1)

def is_cartesian_product(pairs: Set[Tuple[int, int]]) -> Tuple[bool, Optional[Set[int]], Optional[Set[int]]]:
    """Check if a set of pairs can be written as a Cartesian product."""
    if not pairs:
        return True, set(), set()
    
    # Extract all left and right components
    left_components = {pair[0] for pair in pairs}
    right_components = {pair[1] for pair in pairs}
    
    # Check if pairs equals left_components × right_components
    cartesian_product = set(product(left_components, right_components))
    
    if pairs == cartesian_product:
        return True, left_components, right_components
    else:
        return False, None, None

def check_factorization(A: np.ndarray, n1: int, n2: int) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """Check if binary matrix A admits an (n1, n2) factorization."""
    n = A.shape[0]
    if n1 * n2 != n:
        return False, None, None
    
    # Get sparsity pattern
    nonzero_indices = np.where(A == 1)
    sparsity_pattern = list(zip(nonzero_indices[0] + 1, nonzero_indices[1] + 1))  # Convert to 1-based
    
    # Decompose each index pair
    linear_pairs = set()
    for i, j in sparsity_pattern:
        (i1, j1), (i2, j2) = decompose_indices(i, j, n2)
        l1 = linear_index(i1, j1, n1)
        l2 = linear_index(i2, j2, n2)
        linear_pairs.add((l1, l2))
    
    # Check if it's a Cartesian product
    is_factorizable, S1, S2 = is_cartesian_product(linear_pairs)
    
    if not is_factorizable:
        return False, None, None
    
    # Reconstruct factor matrices
    A1 = np.zeros((n1, n1), dtype=int)
    A2 = np.zeros((n2, n2), dtype=int)
    
    if S1 and S2:  # Check if sets are not None
        for l1 in S1:
            i1, j1 = index_to_pair(l1, n1)
            A1[i1-1, j1-1] = 1  # Convert back to 0-based indexing
        
        for l2 in S2:
            i2, j2 = index_to_pair(l2, n2)
            A2[i2-1, j2-1] = 1  # Convert back to 0-based indexing
    
    return True, A1, A2

def find_all_length2_factorizations(A: np.ndarray) -> List[Tuple[int, int]]:
    """Find all length-2 factorizations of matrix A."""
    n = A.shape[0]
    compatible_pairs = get_compatible_pairs(n)
    factorizations = []
    
    for n1, n2 in compatible_pairs:
        is_fact, _, _ = check_factorization(A, n1, n2)
        if is_fact:
            factorizations.append((n1, n2))
    
    return factorizations

def build_factorization_branches(factorizations: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Build factorization branches from length-2 factorizations."""
    if not factorizations:
        return []
    
    # Extract left and right indices
    left_indices = [f[0] for f in factorizations]
    right_indices = [f[1] for f in factorizations]
    
    # Create mapping from left to right
    left_to_right = dict(factorizations)
    
    # Find all unique left indices, sorted
    unique_left = sorted(set(left_indices))
    
    # Remove elements that are multiples of others
    def remove_multiples(indices):
        reduced = []
        for i in indices:
            is_multiple = False
            for j in indices:
                if i != j and i % j == 0:
                    is_multiple = True
                    break
            if not is_multiple:
                reduced.append(i)
        return reduced
    
    roots = remove_multiples(unique_left)
    branches = []
    
    def build_branch_from_root(root):
        branch = [(root, left_to_right[root])]
        current = root
        
        # Find multiples of current that exist in left indices
        while True:
            found_multiple = False
            for left in unique_left:
                if left > current and left % current == 0 and left in left_to_right:
                    # Check if left is not a multiple of any other element between current and left
                    is_direct_multiple = True
                    for other in unique_left:
                        if current < other < left and left % other == 0 and other in left_to_right:
                            is_direct_multiple = False
                            break
                    
                    if is_direct_multiple:
                        branch.append((left, left_to_right[left]))
                        current = left
                        found_multiple = True
                        break
            
            if not found_multiple:
                break
        
        return branch
    
    # Build branches from each root
    for root in roots:
        if root in left_to_right:
            branch = build_branch_from_root(root)
            branches.append(branch)
    
    return branches

class DecompositionGraph:
    """Class to represent and visualize decomposition graphs."""
    
    def __init__(self, factorizations: List[Tuple[int, int]]):
        self.factorizations = factorizations
        self.branches = build_factorization_branches(factorizations)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> nx.MultiDiGraph:
        """Build networkx graph from branches."""
        G = nx.MultiDiGraph()
        
        # Add all left indices as nodes
        left_indices = set(f[0] for f in self.factorizations)
        G.add_nodes_from(left_indices)
        
        # Add edges for each branch
        for i, branch in enumerate(self.branches):
            for j in range(len(branch) - 1):
                left1, right1 = branch[j]
                left2, right2 = branch[j + 1]
                
                # The weight is the factor that relates consecutive left indices
                weight = left2 // left1
                G.add_edge(left1, left2, weight=weight, branch=i, color=i)
        
        return G
    
    def visualize(self, figsize=(10, 6)):
        """Visualize the decomposition graph."""
        plt.figure(figsize=figsize)
        
        if not self.graph.nodes():
            plt.text(0.5, 0.5, 'No factorizations found', 
                    horizontalalignment='center', verticalalignment='center')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title('Decomposition Graph')
            return
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                              node_size=800, alpha=0.7)
        
        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight='bold')
        
        # Draw edges with different colors for different branches
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.branches)))
        
        for i, branch in enumerate(self.branches):
            branch_edges = [(branch[j][0], branch[j+1][0]) for j in range(len(branch)-1)]
            if branch_edges:
                nx.draw_networkx_edges(self.graph, pos, edgelist=branch_edges,
                                     edge_color=[colors[i]], width=2, alpha=0.7,
                                     arrowsize=20)
        
        # Draw edge labels (weights)
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            edge_labels[(u, v)] = str(data['weight'])
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=10)
        
        plt.title('Decomposition Graph')
        plt.axis('off')
        plt.tight_layout()
        
        # Add legend for branches
        if self.branches:
            legend_elements = []
            for i, branch in enumerate(self.branches):
                factorization = self._branch_to_factorization(branch)
                legend_elements.append(plt.Line2D([0], [0], color=colors[i], lw=2,
                                                 label=f'Branch {i+1}: {factorization}'))
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.show()
    
    def _branch_to_factorization(self, branch: List[Tuple[int, int]]) -> Tuple:
        """Convert a branch to its corresponding factorization tuple."""
        if not branch:
            return ()
        
        # Start with the first left index
        factorization = [branch[0][0]]
        
        # Add the weights (factors) between consecutive elements
        for i in range(len(branch) - 1):
            weight = branch[i+1][0] // branch[i][0]
            factorization.append(weight)
        
        # Add the final right index
        factorization.append(branch[-1][1])
        
        return tuple(factorization)
    
    def get_prime_factorizations(self) -> List[Tuple]:
        """Get all prime factorizations from the branches."""
        return [self._branch_to_factorization(branch) for branch in self.branches]

def create_example_matrices() -> Dict[str, np.ndarray]:
    """Create example matrices from the paper."""
    matrices = {}
    
    # Example 2.5 - Matrix with multiple factorizations
    A1 = np.array([[1, 1], [0, 0]])
    A2 = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    matrices['example_2_5'] = np.kron(np.kron(A1, A1), A2)
    
    # Example 3.6 - Identity-like matrix
    matrices['example_3_6_non_factorizable'] = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ])
    
    matrices['example_3_6_factorizable'] = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    # Maximal matrix example (matrix of all ones)
    matrices['maximal_4x4'] = np.ones((4, 4), dtype=int)
    
    # Random sparse matrix for demonstration
    np.random.seed(42)
    sparse_matrix = np.random.choice([0, 1], size=(12, 12), p=[0.8, 0.2])
    matrices['random_sparse'] = sparse_matrix
    
    return matrices

if __name__ == "__main__":
    # Create example matrices
    matrices = create_example_matrices()
    
    print("Kronecker Product Factorization Analysis")
    print("="*50)
    
    # Analyze each matrix
    for name, matrix in matrices.items():
        print(f"\nAnalyzing {name}:")
        print(f"Matrix size: {matrix.shape}")
        print(f"Number of nonzeros: {np.sum(matrix)}")
        
        # Find all length-2 factorizations
        factorizations = find_all_length2_factorizations(matrix)
        print(f"Length-2 factorizations: {factorizations}")
        
        if factorizations:
            # Create decomposition graph
            decomp_graph = DecompositionGraph(factorizations)
            prime_facts = decomp_graph.get_prime_factorizations()
            print(f"Prime factorizations: {prime_facts}")
            
            # Visualize the decomposition graph
            plt.figure(figsize=(12, 8))
            plt.suptitle(f'Decomposition Graph for {name}')
            
            # Show the matrix
            plt.subplot(1, 2, 1)
            plt.imshow(matrix, cmap='Blues', interpolation='nearest')
            plt.title('Binary Matrix')
            plt.colorbar()
            
            # Show decomposition graph
            plt.subplot(1, 2, 2)
            plt.title('Factorization Structure')
            if factorizations:
                decomp_graph.visualize()
            else:
                plt.text(0.5, 0.5, 'No factorizations', ha='center', va='center')
                plt.xlim(0, 1)
                plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.show()
        else:
            print("Matrix is prime (no factorizations found)")
    
    # Demonstrate the algorithm on a specific example
    print("\n" + "="*50)
    print("Detailed Analysis of Example 2.5")
    print("="*50)
    
    A = matrices['example_2_5']
    print(f"Matrix A (size {A.shape}):")
    print(A)
    
    # Check specific factorizations
    compatible_pairs = get_compatible_pairs(A.shape[0])
    print(f"\nCompatible pairs: {compatible_pairs}")
    
    for n1, n2 in compatible_pairs:
        is_fact, A1, A2 = check_factorization(A, n1, n2)
        if is_fact:
            print(f"\n({n1}, {n2}) factorization exists:")
            print(f"A1 (size {A1.shape}):")
            print(A1)
            print(f"A2 (size {A2.shape}):")
            print(A2)
            
            # Verify the factorization
            reconstructed = np.kron(A1, A2)
            print(f"Reconstruction matches: {np.array_equal(A, reconstructed)}")
    
    # Create comprehensive visualization
    plt.figure(figsize=(15, 10))
    
    # Plot original matrix
    plt.subplot(2, 3, 1)
    plt.imshow(A, cmap='Blues', interpolation='nearest')
    plt.title('Original Matrix A')
    plt.colorbar()
    
    # Plot factorizations
    factorizations = find_all_length2_factorizations(A)
    for idx, (n1, n2) in enumerate(factorizations[:4]):  # Show first 4
        is_fact, A1, A2 = check_factorization(A, n1, n2)
        if is_fact:
            plt.subplot(2, 3, idx + 2)
            plt.imshow(np.kron(A1, A2), cmap='Blues', interpolation='nearest')
            plt.title(f'({n1}, {n2}) Factorization')
            plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Show decomposition graph for this example
    decomp_graph = DecompositionGraph(factorizations)
    prime_facts = decomp_graph.get_prime_factorizations()
    
    print(f"\nAll prime factorizations: {prime_facts}")
    
    plt.figure(figsize=(10, 6))
    plt.title('Decomposition Graph for Example 2.5')
    decomp_graph.visualize()
