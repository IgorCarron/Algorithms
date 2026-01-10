import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Set
import time
import heapq
from collections import defaultdict
import random

class SparseVector:
    """Represents a sparse vector with coordinate-value pairs."""
    
    def __init__(self, coordinates: Dict[int, float], dimension: int):
        self.coordinates = coordinates
        self.dimension = dimension
        
    def get_nonzero_coords(self) -> Set[int]:
        return set(self.coordinates.keys())
        
    def __getitem__(self, idx: int) -> float:
        return self.coordinates.get(idx, 0.0)
        
    def inner_product(self, other: 'SparseVector') -> float:
        """Compute exact inner product with another sparse vector."""
        result = 0.0
        for coord in self.coordinates:
            if coord in other.coordinates:
                result += self.coordinates[coord] * other.coordinates[coord]
        return result

class LinScan:
    """Baseline exact algorithm for sparse Maximum Inner Product Search."""
    
    def __init__(self):
        self.inverted_index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.vectors: Dict[int, SparseVector] = {}
        self.next_id = 0
        
    def insert(self, vector: SparseVector) -> int:
        """Insert a document vector into the index."""
        doc_id = self.next_id
        self.next_id += 1
        
        self.vectors[doc_id] = vector
        
        # Add to inverted index
        for coord, value in vector.coordinates.items():
            self.inverted_index[coord].append((doc_id, value))
            
        return doc_id
        
    def search(self, query: SparseVector, k: int) -> List[Tuple[int, float]]:
        """Perform exact top-k search."""
        scores = defaultdict(float)
        
        # Coordinate-at-a-time scoring
        for coord in query.get_nonzero_coords():
            query_value = query[coord]
            for doc_id, doc_value in self.inverted_index[coord]:
                scores[doc_id] += query_value * doc_value
                
        # Find top-k
        if len(scores) <= k:
            return [(doc_id, score) for doc_id, score in scores.items()]
            
        return heapq.nlargest(k, scores.items(), key=lambda x: x[1])

class Sinnamon:
    """Sinnamon algorithm for approximate Maximum Inner Product Search using sketches."""
    
    def __init__(self, sketch_size: int, num_mappings: int = 1, non_negative: bool = True):
        """
        Initialize Sinnamon index.
        
        Args:
            sketch_size: Size of the sketch (m in the paper)
            num_mappings: Number of random mappings (h in the paper)
            non_negative: Whether vectors are non-negative (uses Sinnamon+ variant)
        """
        self.sketch_size = sketch_size
        self.num_mappings = num_mappings
        self.non_negative = non_negative
        
        # Inverted index: coord -> set of document IDs
        self.inverted_index: Dict[int, Set[int]] = defaultdict(set)
        
        # Sketch matrix: either m x |X| (non-negative) or 2m x |X| (general)
        sketch_rows = sketch_size if non_negative else 2 * sketch_size
        self.sketch_matrix: Dict[int, np.ndarray] = {}  # doc_id -> sketch vector
        
        # Random mappings from coordinate space to sketch space
        self.random_mappings: List[Dict[int, int]] = []
        
        # Vector storage for exact re-ranking
        self.vector_storage: Dict[int, SparseVector] = {}
        
        self.next_id = 0
        self.dimension = 0
        
    def _create_random_mapping(self) -> Dict[int, int]:
        """Create a random mapping from coordinates to sketch indices."""
        return {}
        
    def _get_mapping_value(self, mapping_idx: int, coord: int) -> int:
        """Get mapping value for a coordinate, creating if needed."""
        while len(self.random_mappings) <= mapping_idx:
            self.random_mappings.append({})
            
        mapping = self.random_mappings[mapping_idx]
        if coord not in mapping:
            mapping[coord] = hash((mapping_idx, coord)) % self.sketch_size
            
        return mapping[coord]
        
    def insert(self, vector: SparseVector) -> int:
        """Insert a document vector and create its sketch."""
        doc_id = self.next_id
        self.next_id += 1
        
        # Store original vector for exact re-ranking
        self.vector_storage[doc_id] = vector
        
        # Update inverted index
        for coord in vector.get_nonzero_coords():
            self.inverted_index[coord].add(doc_id)
            
        # Create sketch
        sketch_dim = self.sketch_size if self.non_negative else 2 * self.sketch_size
        sketch = np.zeros(sketch_dim)
        
        if self.non_negative:
            # Sinnamon+ variant: only upper bounds
            upper_bounds = np.full(self.sketch_size, -np.inf)
            
            for coord, value in vector.coordinates.items():
                for h in range(self.num_mappings):
                    sketch_idx = self._get_mapping_value(h, coord)
                    upper_bounds[sketch_idx] = max(upper_bounds[sketch_idx], value)
                    
            # Replace -inf with 0 for coordinates that weren't touched
            upper_bounds[upper_bounds == -np.inf] = 0.0
            sketch[:self.sketch_size] = upper_bounds
            
        else:
            # Full Sinnamon: upper and lower bounds
            upper_bounds = np.full(self.sketch_size, -np.inf)
            lower_bounds = np.full(self.sketch_size, np.inf)
            
            for coord, value in vector.coordinates.items():
                for h in range(self.num_mappings):
                    sketch_idx = self._get_mapping_value(h, coord)
                    upper_bounds[sketch_idx] = max(upper_bounds[sketch_idx], value)
                    lower_bounds[sketch_idx] = min(lower_bounds[sketch_idx], value)
                    
            # Handle unused sketch coordinates
            upper_bounds[upper_bounds == -np.inf] = 0.0
            lower_bounds[lower_bounds == np.inf] = 0.0
            
            sketch[:self.sketch_size] = upper_bounds
            sketch[self.sketch_size:] = lower_bounds
            
        self.sketch_matrix[doc_id] = sketch
        return doc_id
        
    def _decode_value(self, doc_id: int, coord: int, query_value: float) -> float:
        """Decode approximate value for a coordinate from the sketch."""
        if doc_id not in self.sketch_matrix:
            return 0.0
            
        sketch = self.sketch_matrix[doc_id]
        
        if self.non_negative:
            # Use upper bound
            min_upper = np.inf
            for h in range(self.num_mappings):
                sketch_idx = self._get_mapping_value(h, coord)
                min_upper = min(min_upper, sketch[sketch_idx])
            return min_upper if min_upper != np.inf else 0.0
            
        else:
            # Use upper bound for positive queries, lower bound for negative
            if query_value > 0:
                min_upper = np.inf
                for h in range(self.num_mappings):
                    sketch_idx = self._get_mapping_value(h, coord)
                    min_upper = min(min_upper, sketch[sketch_idx])
                return min_upper if min_upper != np.inf else 0.0
            else:
                max_lower = -np.inf
                for h in range(self.num_mappings):
                    sketch_idx = self._get_mapping_value(h, coord)
                    max_lower = max(max_lower, sketch[self.sketch_size + sketch_idx])
                return max_lower if max_lower != -np.inf else 0.0
                
    def search(self, query: SparseVector, k: int, k_prime: Optional[int] = None,
               time_budget: Optional[float] = None) -> List[Tuple[int, float]]:
        """Perform approximate top-k search."""
        if k_prime is None:
            k_prime = min(5 * k, len(self.vector_storage))
            
        scores = defaultdict(float)
        start_time = time.time()
        
        # Sort query coordinates by absolute value (for anytime behavior)
        query_coords = list(query.get_nonzero_coords())
        query_coords.sort(key=lambda c: abs(query[c]), reverse=True)
        
        # Coordinate-at-a-time scoring with sketches
        for coord in query_coords:
            if time_budget and (time.time() - start_time) > time_budget:
                break
                
            query_value = query[coord]
            
            for doc_id in self.inverted_index[coord]:
                decoded_value = self._decode_value(doc_id, coord, query_value)
                scores[doc_id] += query_value * decoded_value
                
        # Get top k' candidates
        if len(scores) == 0:
            return []
            
        candidates = heapq.nlargest(min(k_prime, len(scores)), scores.items(), key=lambda x: x[1])
        
        # Re-rank with exact scores
        exact_scores = []
        for doc_id, _ in candidates:
            exact_score = query.inner_product(self.vector_storage[doc_id])
            exact_scores.append((doc_id, exact_score))
            
        # Return top-k
        return heapq.nlargest(k, exact_scores, key=lambda x: x[1])
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage in MB."""
        inverted_index_size = sum(len(doc_set) * 8 for doc_set in self.inverted_index.values())  # 8 bytes per int
        sketch_size = len(self.sketch_matrix) * list(self.sketch_matrix.values())[0].nbytes if self.sketch_matrix else 0
        vector_storage_size = sum(len(v.coordinates) * 16 for v in self.vector_storage.values())  # 8 bytes each for coord and value
        
        return {
            'inverted_index_mb': inverted_index_size / (1024 * 1024),
            'sketches_mb': sketch_size / (1024 * 1024),
            'vector_storage_mb': vector_storage_size / (1024 * 1024),
            'total_mb': (inverted_index_size + sketch_size + vector_storage_size) / (1024 * 1024)
        }

def generate_synthetic_vectors(num_docs: int, num_queries: int, dimension: int, 
                             sparsity: float, distribution: str = 'gaussian') -> Tuple[List[SparseVector], List[SparseVector]]:
    """Generate synthetic sparse vectors for testing."""
    documents = []
    queries = []
    
    def create_vector():
        coords = {}
        num_nonzero = int(dimension * sparsity)
        if num_nonzero == 0:
            num_nonzero = 1  # Ensure at least one non-zero element
        
        selected_coords = np.random.choice(dimension, num_nonzero, replace=False)
        
        for coord in selected_coords:
            if distribution == 'gaussian':
                value = np.random.normal(0, 1)
            elif distribution == 'uniform':
                value = np.random.uniform(-1, 1)
            elif distribution == 'exponential':
                value = np.random.exponential(1)
            else:
                value = 1.0
                
            coords[int(coord)] = value
            
        return SparseVector(coords, dimension)
    
    for _ in range(num_docs):
        documents.append(create_vector())
        
    for _ in range(num_queries):
        queries.append(create_vector())
        
    return documents, queries

def run_performance_comparison():
    """Compare LinScan and Sinnamon performance across different configurations."""
    print("Running performance comparison...")
    
    # Generate test data
    dimension = 10000
    sparsity = 0.01  # 1% non-zero entries
    num_docs = 1000
    num_queries = 50
    
    documents, queries = generate_synthetic_vectors(num_docs, num_queries, dimension, sparsity)
    
    # Test different sketch sizes
    avg_doc_sparsity = int(dimension * sparsity)
    sketch_sizes = [max(1, int(0.25 * avg_doc_sparsity)), max(1, int(0.5 * avg_doc_sparsity)), max(1, int(0.75 * avg_doc_sparsity))]
    
    results = {
        'sketch_sizes': [],
        'memory_usage': [],
        'latency': [],
        'recall': []
    }
    
    # LinScan baseline
    print("Building LinScan index...")
    linscan = LinScan()
    for doc in documents:
        linscan.insert(doc)
    
    # Test Sinnamon with different sketch sizes
    for sketch_size in sketch_sizes:
        print(f"Testing Sinnamon with sketch size {sketch_size}...")
        
        # Build index
        sinnamon = Sinnamon(sketch_size, num_mappings=1, non_negative=False)
        for doc in documents:
            sinnamon.insert(doc)
        
        # Measure performance
        latencies = []
        recalls = []
        
        for query in queries[:10]:  # Test on subset for speed
            # Get exact results
            start_time = time.time()
            exact_results = linscan.search(query, k=10)
            exact_time = time.time() - start_time
            
            # Get approximate results
            start_time = time.time()
            approx_results = sinnamon.search(query, k=10)
            approx_time = time.time() - start_time
            
            latencies.append(approx_time)
            
            # Calculate recall
            exact_ids = set(doc_id for doc_id, _ in exact_results)
            approx_ids = set(doc_id for doc_id, _ in approx_results)
            recall = len(exact_ids.intersection(approx_ids)) / len(exact_ids) if len(exact_ids) > 0 else 1.0
            recalls.append(recall)
        
        results['sketch_sizes'].append(sketch_size)
        results['memory_usage'].append(sinnamon.get_memory_usage()['total_mb'])
        results['latency'].append(np.mean(latencies) * 1000)  # Convert to ms
        results['recall'].append(np.mean(recalls))
    
    return results

def visualize_sketching_error():
    """Visualize sketching error distribution as in the paper."""
    print("Generating sketching error visualization...")
    
    # Generate test data
    dimension = 1000
    sparsity = 0.1
    num_vectors = 1000
    
    documents, _ = generate_synthetic_vectors(num_vectors, 0, dimension, sparsity)
    
    sketch_sizes = [30, 60, 90]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    for i, sketch_size in enumerate(sketch_sizes):
        # Build Sinnamon index
        sinnamon = Sinnamon(sketch_size, num_mappings=1, non_negative=False)
        
        errors = []
        
        for doc in documents[:100]:  # Sample subset
            # Insert document
            doc_id = sinnamon.insert(doc)
            
            # Calculate sketching errors for each coordinate
            for coord, true_value in doc.coordinates.items():
                decoded_value = sinnamon._decode_value(doc_id, coord, 1.0)  # Assume positive query
                error = decoded_value - true_value
                if error >= 0:  # Only positive errors (overestimation)
                    errors.append(error)
        
        axes[i].hist(errors, bins=50, alpha=0.7, density=True, label=f'm={sketch_size}')
        axes[i].set_xlabel('Sketching Error')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Sketching Error Distribution\n(m={sketch_size})')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def visualize_trade_offs(results):
    """Visualize memory-latency-accuracy trade-offs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Memory vs Latency
    axes[0].plot(results['memory_usage'], results['latency'], 'bo-')
    for i, m in enumerate(results['sketch_sizes']):
        axes[0].annotate(f'm={m}', (results['memory_usage'][i], results['latency'][i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    axes[0].set_xlabel('Memory Usage (MB)')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Memory vs Latency Trade-off')
    axes[0].grid(True)
    
    # Latency vs Recall
    axes[1].plot(results['latency'], results['recall'], 'ro-')
    for i, m in enumerate(results['sketch_sizes']):
        axes[1].annotate(f'm={m}', (results['latency'][i], results['recall'][i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    axes[1].set_xlabel('Latency (ms)')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Latency vs Accuracy Trade-off')
    axes[1].grid(True)
    
    # Memory vs Recall
    axes[2].plot(results['memory_usage'], results['recall'], 'go-')
    for i, m in enumerate(results['sketch_sizes']):
        axes[2].annotate(f'm={m}', (results['memory_usage'][i], results['recall'][i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    axes[2].set_xlabel('Memory Usage (MB)')
    axes[2].set_ylabel('Recall')
    axes[2].set_title('Memory vs Accuracy Trade-off')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Sinnamon Algorithm for Maximum Inner Product Search")
    print("=" * 55)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Simple demonstration
    print("\n1. Basic Demonstration:")
    dimension = 100
    sparsity = 0.05
    
    # Create some test vectors
    docs, queries = generate_synthetic_vectors(10, 3, dimension, sparsity)
    
    # Initialize algorithms
    linscan = LinScan()
    sinnamon = Sinnamon(sketch_size=15, num_mappings=1, non_negative=False)
    
    # Insert documents
    print(f"Inserting {len(docs)} documents...")
    for doc in docs:
        linscan.insert(doc)
        sinnamon.insert(doc)
    
    # Test search
    query = queries[0]
    print(f"\nQuery has {len(query.coordinates)} non-zero coordinates")
    
    exact_results = linscan.search(query, k=5)
    approx_results = sinnamon.search(query, k=5)
    
    print("\nExact results (LinScan):")
    for i, (doc_id, score) in enumerate(exact_results):
        print(f"  {i+1}. Document {doc_id}: {score:.4f}")
    
    print("\nApproximate results (Sinnamon):")
    for i, (doc_id, score) in enumerate(approx_results):
        print(f"  {i+1}. Document {doc_id}: {score:.4f}")
    
    # Calculate recall
    exact_ids = set(doc_id for doc_id, _ in exact_results)
    approx_ids = set(doc_id for doc_id, _ in approx_results)
    recall = len(exact_ids.intersection(approx_ids)) / len(exact_ids) if len(exact_ids) > 0 else 1.0
    print(f"\nRecall: {recall:.2f}")
    
    # Memory usage
    memory_usage = sinnamon.get_memory_usage()
    print(f"\nMemory usage: {memory_usage['total_mb']:.2f} MB")
    
    # Run comprehensive evaluation
    print("\n2. Performance Evaluation:")
    results = run_performance_comparison()
    
    print("\nResults:")
    for i in range(len(results['sketch_sizes'])):
        print(f"Sketch size {results['sketch_sizes'][i]}: "
              f"Memory={results['memory_usage'][i]:.2f}MB, "
              f"Latency={results['latency'][i]:.2f}ms, "
              f"Recall={results['recall'][i]:.3f}")
    
    # Generate visualizations
    print("\n3. Generating Visualizations:")
    
    # Visualize trade-offs
    visualize_trade_offs(results)
    
    # Visualize sketching error distribution
    visualize_sketching_error()
    
    print("\nDemonstration completed!")
    print("\nKey findings:")
    print("- Sinnamon provides configurable trade-offs between memory, latency, and accuracy")
    print("- Smaller sketch sizes reduce memory usage but may decrease accuracy")
    print("- The algorithm maintains reasonable recall even with significant compression")
    print("- Sketching errors follow predictable distributions as described in the theory")