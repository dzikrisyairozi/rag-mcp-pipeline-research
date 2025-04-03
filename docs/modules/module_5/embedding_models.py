"""
Embedding Models for RAG Systems

This module provides implementations and comparisons of various embedding models
used in RAG applications. It includes model wrappers, evaluation utilities,
and benchmark functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import time
import numpy as np


class EmbeddingDimension(Enum):
    """Common embedding dimensions for different models."""
    SMALL = 384      # E.g., MiniLM
    MEDIUM = 768     # E.g., BERT base, MPNet
    LARGE = 1024     # E.g., BERT large
    XLARGE = 1536    # E.g., OpenAI ada-002
    XXLARGE = 3072   # E.g., Cohere large


@dataclass
class EmbeddingMetrics:
    """Metrics for evaluating embedding model performance."""
    embedding_time_ms: float      # Average time to generate embeddings in ms
    similarity_calculation_ms: float  # Time to compute similarities in ms
    model_size_mb: float          # Size of the model in MB
    semantic_accuracy: float      # Accuracy on semantic similarity tasks
    retrieval_precision: float    # Precision in retrieval tasks


@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model."""
    name: str
    dimension: int
    normalized: bool = True       # Whether vectors are normalized
    max_sequence_length: int = 512  # Maximum input sequence length
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class EmbeddingModel(ABC):
    """Abstract base class for embedding model implementations."""
    
    def __init__(self, config: EmbeddingModelConfig):
        """Initialize the embedding model with the given configuration."""
        self.config = config
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the model."""
        pass
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Array of embeddings with shape (len(texts), dimension)
        """
        pass
    
    @abstractmethod
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode a list of queries into embeddings.
        Some models use different encoding for queries vs. documents.
        
        Args:
            queries: List of query strings to encode
            
        Returns:
            Array of embeddings with shape (len(queries), dimension)
        """
        pass
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity between a query embedding and document embeddings.
        
        Args:
            query_embedding: Query embedding with shape (dimension,)
            document_embeddings: Document embeddings with shape (num_docs, dimension)
            
        Returns:
            Array of similarity scores with shape (num_docs,)
        """
        # Default to cosine similarity
        if self.config.normalized:
            # For normalized vectors, dot product is equivalent to cosine similarity
            return np.dot(document_embeddings, query_embedding)
        else:
            # Compute cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(document_embeddings, axis=1)
            return np.dot(document_embeddings, query_embedding) / (query_norm * doc_norms)
    
    def benchmark(self, texts: List[str], queries: List[str]) -> EmbeddingMetrics:
        """
        Benchmark the embedding model on a dataset.
        
        Args:
            texts: List of documents to encode
            queries: List of queries to encode
            
        Returns:
            EmbeddingMetrics with performance metrics
        """
        # Measure document encoding time
        start_time = time.time()
        doc_embeddings = self.encode(texts)
        doc_time = time.time() - start_time
        doc_time_ms = doc_time * 1000 / len(texts)
        
        # Measure query encoding time
        start_time = time.time()
        query_embeddings = self.encode_queries(queries)
        query_time = time.time() - start_time
        query_time_ms = query_time * 1000 / len(queries)
        
        # Average encoding time
        avg_embedding_time = (doc_time_ms + query_time_ms) / 2
        
        # Measure similarity calculation time
        start_time = time.time()
        for query_emb in query_embeddings:
            _ = self.compute_similarity(query_emb, doc_embeddings)
        sim_time = time.time() - start_time
        sim_time_ms = sim_time * 1000 / len(queries)
        
        # These would be calculated from actual evaluations in real implementation
        # Using placeholders here
        semantic_accuracy = 0.8  # Placeholder
        retrieval_precision = 0.75  # Placeholder
        model_size_mb = 100  # Placeholder
        
        return EmbeddingMetrics(
            embedding_time_ms=avg_embedding_time,
            similarity_calculation_ms=sim_time_ms,
            model_size_mb=model_size_mb,
            semantic_accuracy=semantic_accuracy,
            retrieval_precision=retrieval_precision
        )


class SentenceTransformerModel(EmbeddingModel):
    """Wrapper for sentence-transformer models."""
    
    def initialize(self) -> bool:
        """Initialize the sentence-transformer model."""
        try:
            # In a real implementation, this would load the model
            print(f"Loading sentence-transformer model: {self.config.name}")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence-transformer."""
        if not self.is_initialized:
            print("Model not initialized.")
            return np.array([])
        
        # Mock encoding implementation
        print(f"Encoding {len(texts)} documents with {self.config.name}")
        # Generate random embeddings for demonstration
        embeddings = np.random.rand(len(texts), self.config.dimension)
        
        # Normalize if specified in config
        if self.config.normalized:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
        return embeddings
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries using sentence-transformer."""
        # For most sentence transformers, query encoding is the same as document encoding
        return self.encode(queries)


class OpenAIEmbeddingModel(EmbeddingModel):
    """Wrapper for OpenAI embedding models."""
    
    def initialize(self) -> bool:
        """Initialize the OpenAI API client."""
        try:
            # In a real implementation, this would set up the OpenAI API client
            print(f"Setting up OpenAI API client for model: {self.config.name}")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to set up OpenAI API: {e}")
            return False
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI embeddings API."""
        if not self.is_initialized:
            print("Model not initialized.")
            return np.array([])
        
        # Mock encoding implementation
        print(f"Encoding {len(texts)} documents with OpenAI {self.config.name}")
        # Generate random embeddings for demonstration
        embeddings = np.random.rand(len(texts), self.config.dimension)
        
        # Normalize if specified in config
        if self.config.normalized:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
        return embeddings
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries using OpenAI embeddings API."""
        # For OpenAI embeddings, query encoding is the same as document encoding
        return self.encode(queries)


class CohereEmbeddingModel(EmbeddingModel):
    """Wrapper for Cohere embedding models."""
    
    def initialize(self) -> bool:
        """Initialize the Cohere API client."""
        try:
            # In a real implementation, this would set up the Cohere API client
            print(f"Setting up Cohere API client for model: {self.config.name}")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to set up Cohere API: {e}")
            return False
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Cohere embeddings API."""
        if not self.is_initialized:
            print("Model not initialized.")
            return np.array([])
        
        # Mock encoding implementation
        print(f"Encoding {len(texts)} documents with Cohere {self.config.name}")
        # Generate random embeddings for demonstration
        embeddings = np.random.rand(len(texts), self.config.dimension)
        
        # Normalize if specified in config
        if self.config.normalized:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
        return embeddings
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries using Cohere embeddings API."""
        # For Cohere embeddings, query encoding is the same as document encoding
        return self.encode(queries)


class E5EmbeddingModel(EmbeddingModel):
    """Wrapper for E5 embedding models (different encoders for queries and documents)."""
    
    def initialize(self) -> bool:
        """Initialize the E5 model."""
        try:
            # In a real implementation, this would load the model
            print(f"Loading E5 model: {self.config.name}")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using E5 document encoder."""
        if not self.is_initialized:
            print("Model not initialized.")
            return np.array([])
        
        # E5 prepends "passage: " to documents
        texts = [f"passage: {text}" for text in texts]
        
        # Mock encoding implementation
        print(f"Encoding {len(texts)} documents with E5 {self.config.name}")
        # Generate random embeddings for demonstration
        embeddings = np.random.rand(len(texts), self.config.dimension)
        
        # Normalize if specified in config
        if self.config.normalized:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
        return embeddings
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries using E5 query encoder."""
        if not self.is_initialized:
            print("Model not initialized.")
            return np.array([])
        
        # E5 prepends "query: " to queries
        queries = [f"query: {query}" for query in queries]
        
        # Mock encoding implementation
        print(f"Encoding {len(queries)} queries with E5 {self.config.name}")
        # Generate random embeddings for demonstration
        embeddings = np.random.rand(len(queries), self.config.dimension)
        
        # Normalize if specified in config
        if self.config.normalized:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
        return embeddings


def compare_embedding_models(texts: List[str], queries: List[str]) -> Dict[str, EmbeddingMetrics]:
    """
    Compare different embedding models on the same dataset.
    
    Args:
        texts: List of documents to encode
        queries: List of queries to encode
        
    Returns:
        Dictionary mapping model names to their performance metrics
    """
    # Configure models
    model_configs = [
        EmbeddingModelConfig(name="sentence-transformers/all-MiniLM-L6-v2", 
                            dimension=EmbeddingDimension.SMALL.value),
        EmbeddingModelConfig(name="sentence-transformers/all-mpnet-base-v2",
                            dimension=EmbeddingDimension.MEDIUM.value),
        EmbeddingModelConfig(name="text-embedding-ada-002",
                            dimension=EmbeddingDimension.XLARGE.value),
        EmbeddingModelConfig(name="cohere-embed-multilingual-v3.0",
                            dimension=EmbeddingDimension.XLARGE.value),
        EmbeddingModelConfig(name="intfloat/e5-large-v2",
                            dimension=EmbeddingDimension.LARGE.value),
    ]
    
    # Initialize models
    models = {
        "MiniLM": SentenceTransformerModel(model_configs[0]),
        "MPNet": SentenceTransformerModel(model_configs[1]),
        "OpenAI": OpenAIEmbeddingModel(model_configs[2]),
        "Cohere": CohereEmbeddingModel(model_configs[3]),
        "E5": E5EmbeddingModel(model_configs[4]),
    }
    
    # Run benchmarks
    results = {}
    for name, model in models.items():
        print(f"\nBenchmarking {name}...")
        model.initialize()
        metrics = model.benchmark(texts, queries)
        results[name] = metrics
        print(f"Results for {name}:")
        print(f"  Embedding time: {metrics.embedding_time_ms:.2f} ms")
        print(f"  Similarity calculation time: {metrics.similarity_calculation_ms:.2f} ms")
        print(f"  Semantic accuracy: {metrics.semantic_accuracy:.4f}")
        print(f"  Retrieval precision: {metrics.retrieval_precision:.4f}")
    
    return results


def print_comparison_table(results: Dict[str, EmbeddingMetrics]):
    """Print a formatted comparison table of embedding model performance."""
    print("\n" + "=" * 100)
    print(f"{'Model':<15} | {'Embed Time (ms)':<15} | {'Sim Calc (ms)':<15} | {'Semantic Acc':<15} | {'Retrieval Prec':<15}")
    print("-" * 100)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} | {metrics.embedding_time_ms:<15.2f} | "
              f"{metrics.similarity_calculation_ms:<15.2f} | {metrics.semantic_accuracy:<15.4f} | "
              f"{metrics.retrieval_precision:<15.4f}")
    
    print("=" * 100)


def get_recommended_model(results: Dict[str, EmbeddingMetrics], 
                         prioritize_speed: bool = False) -> str:
    """
    Get recommended model based on performance metrics.
    
    Args:
        results: Dictionary of model metrics
        prioritize_speed: Whether to prioritize encoding speed over accuracy
        
    Returns:
        Name of the recommended model
    """
    if prioritize_speed:
        # Sort by embedding time
        return min(results.items(), key=lambda x: x[1].embedding_time_ms)[0]
    else:
        # Use a weighted combination of semantic accuracy and retrieval precision
        return max(results.items(), 
                   key=lambda x: 0.5 * x[1].semantic_accuracy + 0.5 * x[1].retrieval_precision)[0]


class EmbeddingModelFactory:
    """Factory class for creating embedding models."""
    
    @staticmethod
    def create_model(model_name: str) -> Optional[EmbeddingModel]:
        """
        Create an embedding model based on the model name.
        
        Args:
            model_name: Name of the model to create
            
        Returns:
            An initialized embedding model or None if not supported
        """
        # Define configurations for supported models
        model_configs = {
            "miniLM": EmbeddingModelConfig(
                name="sentence-transformers/all-MiniLM-L6-v2",
                dimension=EmbeddingDimension.SMALL.value
            ),
            "mpnet": EmbeddingModelConfig(
                name="sentence-transformers/all-mpnet-base-v2",
                dimension=EmbeddingDimension.MEDIUM.value
            ),
            "openai": EmbeddingModelConfig(
                name="text-embedding-ada-002",
                dimension=EmbeddingDimension.XLARGE.value
            ),
            "cohere": EmbeddingModelConfig(
                name="cohere-embed-multilingual-v3.0",
                dimension=EmbeddingDimension.XLARGE.value
            ),
            "e5": EmbeddingModelConfig(
                name="intfloat/e5-large-v2",
                dimension=EmbeddingDimension.LARGE.value
            ),
        }
        
        # Create the appropriate model based on the name
        lower_name = model_name.lower()
        if lower_name in ["minilm", "all-minilm", "all-minilm-l6-v2"]:
            model = SentenceTransformerModel(model_configs["miniLM"])
        elif lower_name in ["mpnet", "all-mpnet", "all-mpnet-base-v2"]:
            model = SentenceTransformerModel(model_configs["mpnet"])
        elif lower_name in ["openai", "ada", "ada-002", "text-embedding-ada-002"]:
            model = OpenAIEmbeddingModel(model_configs["openai"])
        elif lower_name in ["cohere", "cohere-embed", "cohere-multilingual"]:
            model = CohereEmbeddingModel(model_configs["cohere"])
        elif lower_name in ["e5", "e5-large", "e5-large-v2"]:
            model = E5EmbeddingModel(model_configs["e5"])
        else:
            print(f"Model {model_name} not supported.")
            return None
        
        # Initialize the model
        model.initialize()
        return model


if __name__ == "__main__":
    print("Embedding Models for RAG Systems")
    print("--------------------------------")
    
    # Sample data for benchmarking
    sample_texts = [
        "Retrieval-Augmented Generation (RAG) is a technique that enhances language models with external knowledge.",
        "Vector databases store embeddings for efficient similarity search.",
        "Embedding models convert text into numerical vectors that capture semantic meaning.",
        "Chunking strategies affect the granularity of information retrieval in RAG systems.",
        "Hybrid retrieval combines dense and sparse methods for better search results.",
    ]
    
    sample_queries = [
        "What is RAG?",
        "How do vector databases work?",
        "What are text embeddings?",
        "How should I split my documents?",
        "What is hybrid search?",
    ]
    
    # Run comparison benchmark
    results = compare_embedding_models(sample_texts, sample_queries)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Get recommendations
    speed_recommendation = get_recommended_model(results, prioritize_speed=True)
    accuracy_recommendation = get_recommended_model(results, prioritize_speed=False)
    
    print(f"\nRecommendations:")
    print(f"For speed-critical applications: {speed_recommendation}")
    print(f"For accuracy-critical applications: {accuracy_recommendation}")
    
    print("\nExample usage with the factory pattern:")
    model = EmbeddingModelFactory.create_model("mpnet")
    if model:
        embeddings = model.encode(["This is a test document."])
        print(f"Generated embedding with shape: {embeddings.shape}")
    
    print("\nNote: These are simulated results. In real-world applications,")
    print("performance will vary based on hardware, model versions,")
    print("and specific implementation details.") 