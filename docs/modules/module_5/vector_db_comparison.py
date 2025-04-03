"""
Vector Database Comparison for RAG Systems

This module provides a comparative analysis of popular vector databases used in RAG applications.
It includes benchmarking utilities, configuration classes, and example implementations
for different vector database solutions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np


class IndexType(Enum):
    """Types of indexes supported by vector databases."""
    FLAT = "flat"  # Exact but slow for large datasets
    IVF = "ivf"    # Inverted file index, faster but approximate
    HNSW = "hnsw"  # Hierarchical Navigable Small World, good balance of speed/accuracy
    PQ = "pq"      # Product Quantization, compressed vectors
    CUSTOM = "custom"  # Database-specific indexing


@dataclass
class VectorDBMetrics:
    """Metrics for evaluating vector database performance."""
    query_latency_ms: float  # Average query time in milliseconds
    memory_usage_mb: float   # Memory usage in megabytes
    index_build_time_s: float  # Time to build the index in seconds
    recall_at_10: float      # Recall@10 metric (% of relevant results in top 10)
    throughput_qps: float    # Queries per second


@dataclass
class VectorDBConfig:
    """Configuration for a vector database."""
    name: str
    vector_dim: int
    index_type: IndexType
    distance_metric: str  # "cosine", "euclidean", "dot"
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class VectorDatabase(ABC):
    """Abstract base class for vector database implementations."""
    
    def __init__(self, config: VectorDBConfig):
        """Initialize the vector database with the given configuration."""
        self.config = config
        self.is_initialized = False
        
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    def create_index(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Create a search index from the provided vectors and metadata."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Add new vectors to the database."""
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from the database."""
        pass
    
    def benchmark(self, query_vectors: np.ndarray, ground_truth: List[List[int]], 
                  num_queries: int = 100) -> VectorDBMetrics:
        """Run benchmarks on the vector database."""
        # Time query latency
        start_time = time.time()
        for i in range(min(num_queries, len(query_vectors))):
            _ = self.search(query_vectors[i], top_k=10)
        end_time = time.time()
        query_time = (end_time - start_time) * 1000 / min(num_queries, len(query_vectors))
        
        # Calculate throughput
        throughput = 1000 / query_time if query_time > 0 else 0
        
        # Compute recall@10
        recall_sum = 0
        for i in range(min(num_queries, len(query_vectors))):
            results = self.search(query_vectors[i], top_k=10)
            result_ids = [r.get("id") for r in results]
            relevant_count = len(set(result_ids).intersection(set(ground_truth[i])))
            recall = relevant_count / len(ground_truth[i]) if ground_truth[i] else 0
            recall_sum += recall
        
        avg_recall = recall_sum / min(num_queries, len(query_vectors))
        
        # Placeholder for memory usage (implementation-dependent)
        memory_usage = 0  # Would be populated by actual database monitoring
        
        # Placeholder for index build time (captured during create_index)
        index_build_time = 0  # Would be populated during index creation
        
        return VectorDBMetrics(
            query_latency_ms=query_time,
            memory_usage_mb=memory_usage,
            index_build_time_s=index_build_time,
            recall_at_10=avg_recall,
            throughput_qps=throughput
        )


class PineconeDB(VectorDatabase):
    """Implementation for Pinecone vector database."""
    
    def connect(self) -> bool:
        """Connect to Pinecone."""
        try:
            # In a real implementation, this would use the Pinecone client
            print(f"Connecting to Pinecone with {self.config.name}")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to connect to Pinecone: {e}")
            return False
    
    def create_index(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Create a Pinecone index."""
        if not self.is_initialized:
            print("Database not initialized.")
            return False
        
        try:
            # Mock implementation of index creation
            print(f"Creating {self.config.index_type.value} index in Pinecone")
            print(f"Indexing {len(vectors)} vectors of dimension {self.config.vector_dim}")
            return True
        except Exception as e:
            print(f"Failed to create index: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone."""
        if not self.is_initialized:
            print("Database not initialized.")
            return []
        
        # Mock search implementation
        return [{"id": f"doc_{i}", "score": 0.9 - (i * 0.05), "metadata": {}} for i in range(top_k)]
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to Pinecone index."""
        if not self.is_initialized:
            print("Database not initialized.")
            return False
        
        try:
            print(f"Adding {len(vectors)} vectors to Pinecone index")
            return True
        except Exception as e:
            print(f"Failed to add vectors: {e}")
            return False
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Pinecone index."""
        if not self.is_initialized:
            print("Database not initialized.")
            return False
        
        try:
            print(f"Deleting {len(ids)} vectors from Pinecone index")
            return True
        except Exception as e:
            print(f"Failed to delete vectors: {e}")
            return False


class ChromaDB(VectorDatabase):
    """Implementation for Chroma vector database."""
    
    def connect(self) -> bool:
        """Connect to ChromaDB."""
        try:
            # In a real implementation, this would initialize the ChromaDB client
            print(f"Initializing ChromaDB with {self.config.name}")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize ChromaDB: {e}")
            return False
    
    def create_index(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Create a ChromaDB collection."""
        if not self.is_initialized:
            print("Database not initialized.")
            return False
        
        try:
            # Mock implementation of collection creation
            print(f"Creating ChromaDB collection with {len(vectors)} documents")
            return True
        except Exception as e:
            print(f"Failed to create collection: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors in ChromaDB."""
        if not self.is_initialized:
            print("Database not initialized.")
            return []
        
        # Mock search implementation
        return [{"id": f"doc_{i}", "score": 0.95 - (i * 0.03), "metadata": {}} for i in range(top_k)]
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to ChromaDB collection."""
        if not self.is_initialized:
            print("Database not initialized.")
            return False
        
        try:
            print(f"Adding {len(vectors)} documents to ChromaDB collection")
            return True
        except Exception as e:
            print(f"Failed to add documents: {e}")
            return False
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from ChromaDB collection."""
        if not self.is_initialized:
            print("Database not initialized.")
            return False
        
        try:
            print(f"Deleting {len(ids)} documents from ChromaDB collection")
            return True
        except Exception as e:
            print(f"Failed to delete documents: {e}")
            return False


class WeaviateDB(VectorDatabase):
    """Implementation for Weaviate vector database."""
    
    def connect(self) -> bool:
        """Connect to Weaviate."""
        try:
            # In a real implementation, this would use the Weaviate client
            print(f"Connecting to Weaviate with {self.config.name}")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to connect to Weaviate: {e}")
            return False
    
    def create_index(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Create a Weaviate class/schema."""
        if not self.is_initialized:
            print("Database not initialized.")
            return False
        
        try:
            # Mock implementation of schema creation
            print(f"Creating Weaviate schema with {self.config.index_type.value} index")
            return True
        except Exception as e:
            print(f"Failed to create schema: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors in Weaviate."""
        if not self.is_initialized:
            print("Database not initialized.")
            return []
        
        # Mock search implementation
        return [{"id": f"doc_{i}", "score": 0.92 - (i * 0.04), "metadata": {}} for i in range(top_k)]
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to Weaviate."""
        if not self.is_initialized:
            print("Database not initialized.")
            return False
        
        try:
            print(f"Adding {len(vectors)} objects to Weaviate")
            return True
        except Exception as e:
            print(f"Failed to add objects: {e}")
            return False
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Weaviate."""
        if not self.is_initialized:
            print("Database not initialized.")
            return False
        
        try:
            print(f"Deleting {len(ids)} objects from Weaviate")
            return True
        except Exception as e:
            print(f"Failed to delete objects: {e}")
            return False


class QdrantDB(VectorDatabase):
    """Implementation for Qdrant vector database."""
    
    def connect(self) -> bool:
        """Connect to Qdrant."""
        try:
            # In a real implementation, this would use the Qdrant client
            print(f"Connecting to Qdrant with {self.config.name}")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")
            return False
    
    def create_index(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Create a Qdrant collection."""
        if not self.is_initialized:
            print("Database not initialized.")
            return False
        
        try:
            # Mock implementation of collection creation
            print(f"Creating Qdrant collection with {self.config.index_type.value} index")
            print(f"Vectors dimension: {self.config.vector_dim}")
            return True
        except Exception as e:
            print(f"Failed to create collection: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant."""
        if not self.is_initialized:
            print("Database not initialized.")
            return []
        
        # Mock search implementation
        return [{"id": f"doc_{i}", "score": 0.93 - (i * 0.04), "metadata": {}} for i in range(top_k)]
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to Qdrant collection."""
        if not self.is_initialized:
            print("Database not initialized.")
            return False
        
        try:
            print(f"Adding {len(vectors)} points to Qdrant collection")
            return True
        except Exception as e:
            print(f"Failed to add points: {e}")
            return False
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Qdrant collection."""
        if not self.is_initialized:
            print("Database not initialized.")
            return False
        
        try:
            print(f"Deleting {len(ids)} points from Qdrant collection")
            return True
        except Exception as e:
            print(f"Failed to delete points: {e}")
            return False


def compare_vector_dbs(dataset_size: int = 10000, vector_dim: int = 768, 
                      num_queries: int = 100) -> Dict[str, VectorDBMetrics]:
    """
    Compare different vector databases using the same dataset and queries.
    
    Args:
        dataset_size: Number of vectors in the test dataset
        vector_dim: Dimension of the vectors
        num_queries: Number of queries to run for benchmarking
        
    Returns:
        Dictionary mapping database names to their performance metrics
    """
    # Generate random test data
    np.random.seed(42)  # For reproducibility
    vectors = np.random.rand(dataset_size, vector_dim).astype(np.float32)
    metadata = [{"text": f"Document {i}", "category": f"cat_{i % 10}"} for i in range(dataset_size)]
    
    # Generate query vectors and mock ground truth
    query_vectors = np.random.rand(num_queries, vector_dim).astype(np.float32)
    ground_truth = [[j for j in range(i % 10, i % 10 + 10)] for i in range(num_queries)]
    
    # Configure databases
    db_configs = [
        VectorDBConfig(name="pinecone", vector_dim=vector_dim, 
                      index_type=IndexType.HNSW, distance_metric="cosine"),
        VectorDBConfig(name="chroma", vector_dim=vector_dim,
                      index_type=IndexType.HNSW, distance_metric="cosine"),
        VectorDBConfig(name="weaviate", vector_dim=vector_dim,
                      index_type=IndexType.HNSW, distance_metric="cosine"),
        VectorDBConfig(name="qdrant", vector_dim=vector_dim,
                      index_type=IndexType.HNSW, distance_metric="cosine"),
    ]
    
    # Initialize databases
    dbs = {
        "pinecone": PineconeDB(db_configs[0]),
        "chroma": ChromaDB(db_configs[1]),
        "weaviate": WeaviateDB(db_configs[2]),
        "qdrant": QdrantDB(db_configs[3]),
    }
    
    # Run benchmarks
    results = {}
    for name, db in dbs.items():
        print(f"\nBenchmarking {name}...")
        db.connect()
        db.create_index(vectors, metadata)
        metrics = db.benchmark(query_vectors, ground_truth, num_queries)
        results[name] = metrics
        print(f"Results for {name}:")
        print(f"  Query latency: {metrics.query_latency_ms:.2f} ms")
        print(f"  Throughput: {metrics.throughput_qps:.2f} queries/second")
        print(f"  Recall@10: {metrics.recall_at_10:.4f}")
    
    return results


def print_comparison_table(results: Dict[str, VectorDBMetrics]):
    """Print a formatted comparison table of vector database performance."""
    print("\n" + "=" * 80)
    print(f"{'Database':<15} | {'Latency (ms)':<15} | {'Throughput (QPS)':<20} | {'Recall@10':<15}")
    print("-" * 80)
    
    for db_name, metrics in results.items():
        print(f"{db_name:<15} | {metrics.query_latency_ms:<15.2f} | {metrics.throughput_qps:<20.2f} | {metrics.recall_at_10:<15.4f}")
    
    print("=" * 80)


def get_recommended_db(results: Dict[str, VectorDBMetrics], 
                      prioritize_speed: bool = False) -> str:
    """
    Get recommended database based on performance metrics.
    
    Args:
        results: Dictionary of database metrics
        prioritize_speed: Whether to prioritize query speed over recall
        
    Returns:
        Name of the recommended database
    """
    if prioritize_speed:
        return max(results.items(), key=lambda x: x[1].throughput_qps)[0]
    else:
        return max(results.items(), key=lambda x: x[1].recall_at_10)[0]


if __name__ == "__main__":
    print("Vector Database Comparison for RAG Systems")
    print("------------------------------------------")
    
    # Run comparison benchmark
    results = compare_vector_dbs(dataset_size=10000, vector_dim=768, num_queries=100)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Get recommendations
    speed_recommendation = get_recommended_db(results, prioritize_speed=True)
    accuracy_recommendation = get_recommended_db(results, prioritize_speed=False)
    
    print(f"\nRecommendations:")
    print(f"For speed-critical applications: {speed_recommendation}")
    print(f"For accuracy-critical applications: {accuracy_recommendation}")
    
    print("\nNote: These are simulated results. In real-world applications,")
    print("performance will vary based on hardware, network conditions,")
    print("dataset characteristics, and specific configuration settings.") 