"""
Hybrid Retrieval for RAG Systems

This module provides implementations of hybrid retrieval methods that combine
dense vector retrieval with sparse retrieval techniques for improved results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import time
import re


@dataclass
class Document:
    """Represents a document in the retrieval system."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    score: float = 0.0


@dataclass
class SearchResult:
    """Represents a search result with its relevance score."""
    document: Document
    score: float
    rank: int
    method: str = "hybrid"  # The method that produced this result


class RetrievalMethod(ABC):
    """Abstract base class for retrieval methods."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for relevant documents based on the query.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of search results
        """
        pass


class DenseRetrieval(RetrievalMethod):
    """
    Dense retrieval using vector embeddings.
    Finds similar documents based on semantic meaning using vector similarity.
    """
    
    def __init__(self, documents: List[Document], embedding_model: Any = None):
        """
        Initialize the dense retrieval method.
        
        Args:
            documents: List of documents with embeddings
            embedding_model: Model to create embeddings for queries
        """
        self.documents = documents
        self.embedding_model = embedding_model
        
        # Ensure all documents have embeddings
        self._check_embeddings()
    
    def _check_embeddings(self):
        """Check if documents have embeddings and create if missing."""
        missing_embeddings = 0
        
        for doc in self.documents:
            if doc.embedding is None:
                missing_embeddings += 1
        
        if missing_embeddings > 0:
            print(f"Warning: {missing_embeddings} documents are missing embeddings.")
            if self.embedding_model:
                print("Generating missing embeddings...")
                self._generate_embeddings()
            else:
                print("No embedding model provided, cannot generate embeddings.")
                print("Dense retrieval will not work correctly.")
    
    def _generate_embeddings(self):
        """Generate embeddings for documents that don't have them."""
        # In a real implementation, this would use the embedding model
        # Here we just generate random embeddings for demonstration
        dim = 768  # Typical embedding dimension
        
        for i, doc in enumerate(self.documents):
            if doc.embedding is None:
                doc.embedding = np.random.rand(dim)
                # Normalize embedding
                doc.embedding = doc.embedding / np.linalg.norm(doc.embedding)
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for documents similar to the query using vector similarity."""
        print(f"Dense retrieval for query: {query}")
        
        # Generate query embedding
        if self.embedding_model:
            # In a real implementation, this would use the model
            query_embedding = np.random.rand(768)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        else:
            # Generate random embedding for demonstration
            query_embedding = np.random.rand(768)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate similarity scores
        scores = []
        for doc in self.documents:
            if doc.embedding is not None:
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc.embedding)
                scores.append((doc, similarity))
        
        # Sort by similarity score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create search results
        results = []
        for i, (doc, score) in enumerate(scores[:top_k]):
            results.append(SearchResult(
                document=doc,
                score=score,
                rank=i + 1,
                method="dense"
            ))
        
        return results


class SparseRetrieval(RetrievalMethod):
    """
    Sparse retrieval using keyword matching (BM25 or TF-IDF).
    Finds documents containing the same terms as the query.
    """
    
    def __init__(self, documents: List[Document], method: str = "bm25"):
        """
        Initialize the sparse retrieval method.
        
        Args:
            documents: List of documents
            method: Sparse retrieval method ("bm25" or "tfidf")
        """
        self.documents = documents
        self.method = method.lower()
        
        # Index the documents
        self._create_index()
    
    def _create_index(self):
        """Create the sparse index for retrieval."""
        print(f"Creating {self.method} index for {len(self.documents)} documents")
        
        # In a real implementation, this would create a proper BM25 or TF-IDF index
        # For demonstration, we just tokenize the documents
        
        # Create document term frequency maps
        self.doc_term_freqs = []
        self.term_doc_freqs = {}  # Term to document count
        
        for doc in self.documents:
            # Simple tokenization and counting
            tokens = self._tokenize(doc.content)
            term_freq = {}
            
            for token in tokens:
                if token in term_freq:
                    term_freq[token] += 1
                else:
                    term_freq[token] = 1
                    
                    # Update document frequency
                    if token in self.term_doc_freqs:
                        self.term_doc_freqs[token] += 1
                    else:
                        self.term_doc_freqs[token] = 1
            
            self.doc_term_freqs.append((doc, term_freq))
        
        # For BM25, calculate average document length
        if self.method == "bm25":
            self.avg_doc_length = sum(len(self._tokenize(doc.content)) for doc in self.documents) / len(self.documents)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization function."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Split by whitespace
        return text.split()
    
    def _calculate_bm25(self, query_terms: List[str], doc_idx: int, term_freq: Dict[str, int]) -> float:
        """Calculate BM25 score for a document."""
        # BM25 parameters
        k1 = 1.2
        b = 0.75
        
        doc = self.documents[doc_idx]
        doc_length = len(self._tokenize(doc.content))
        score = 0.0
        
        for term in query_terms:
            if term in term_freq:
                # Term frequency in document
                tf = term_freq[term]
                
                # Inverse document frequency
                idf = np.log(1 + (len(self.documents) - self.term_doc_freqs.get(term, 0) + 0.5) / 
                            (self.term_doc_freqs.get(term, 0) + 0.5))
                
                # BM25 score for term
                term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / self.avg_doc_length))
                score += term_score
        
        return score
    
    def _calculate_tfidf(self, query_terms: List[str], doc_idx: int, term_freq: Dict[str, int]) -> float:
        """Calculate TF-IDF score for a document."""
        doc = self.documents[doc_idx]
        score = 0.0
        
        for term in query_terms:
            if term in term_freq:
                # Term frequency in document
                tf = term_freq[term]
                
                # Inverse document frequency
                idf = np.log(len(self.documents) / (1 + self.term_doc_freqs.get(term, 0)))
                
                # TF-IDF score for term
                score += tf * idf
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for documents containing query terms."""
        print(f"Sparse retrieval ({self.method}) for query: {query}")
        
        # Tokenize query
        query_terms = self._tokenize(query)
        
        # Calculate scores for each document
        scores = []
        for i, (doc, term_freq) in enumerate(self.doc_term_freqs):
            if self.method == "bm25":
                score = self._calculate_bm25(query_terms, i, term_freq)
            else:  # tfidf
                score = self._calculate_tfidf(query_terms, i, term_freq)
            
            scores.append((doc, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create search results
        results = []
        for i, (doc, score) in enumerate(scores[:top_k]):
            results.append(SearchResult(
                document=doc,
                score=score,
                rank=i + 1,
                method="sparse"
            ))
        
        return results


class HybridRetrieval(RetrievalMethod):
    """
    Hybrid retrieval combining dense and sparse methods.
    Provides better recall and precision than either method alone.
    """
    
    def __init__(self, 
                dense_retriever: DenseRetrieval,
                sparse_retriever: SparseRetrieval,
                alpha: float = 0.5):
        """
        Initialize the hybrid retrieval method.
        
        Args:
            dense_retriever: Dense retrieval method
            sparse_retriever: Sparse retrieval method
            alpha: Weight for dense retrieval (1-alpha for sparse)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using both dense and sparse methods and combine results."""
        print(f"Hybrid retrieval for query: {query}")
        
        # Get results from both methods
        # Retrieve more results than needed for better coverage
        dense_results = self.dense_retriever.search(query, top_k=top_k * 2)
        sparse_results = self.sparse_retriever.search(query, top_k=top_k * 2)
        
        # Normalize scores
        self._normalize_scores(dense_results, sparse_results)
        
        # Combine results
        combined_results = self._combine_results(dense_results, sparse_results, top_k)
        
        return combined_results
    
    def _normalize_scores(self, dense_results: List[SearchResult], 
                          sparse_results: List[SearchResult]):
        """Normalize scores to be in the range [0, 1]."""
        # Find max scores
        max_dense = max([r.score for r in dense_results]) if dense_results else 1.0
        max_sparse = max([r.score for r in sparse_results]) if sparse_results else 1.0
        
        # Normalize dense scores
        for result in dense_results:
            result.score = result.score / max_dense if max_dense > 0 else 0
        
        # Normalize sparse scores
        for result in sparse_results:
            result.score = result.score / max_sparse if max_sparse > 0 else 0
    
    def _combine_results(self, 
                        dense_results: List[SearchResult],
                        sparse_results: List[SearchResult],
                        top_k: int) -> List[SearchResult]:
        """Combine results from both methods."""
        # Create a mapping from document ID to results
        result_map = {}
        
        # Add dense results
        for result in dense_results:
            result_map[result.document.id] = {
                "document": result.document,
                "dense_score": result.score,
                "sparse_score": 0.0
            }
        
        # Add or update with sparse results
        for result in sparse_results:
            if result.document.id in result_map:
                result_map[result.document.id]["sparse_score"] = result.score
            else:
                result_map[result.document.id] = {
                    "document": result.document,
                    "dense_score": 0.0,
                    "sparse_score": result.score
                }
        
        # Calculate combined scores
        combined_results = []
        for doc_id, data in result_map.items():
            combined_score = self.alpha * data["dense_score"] + (1 - self.alpha) * data["sparse_score"]
            
            combined_results.append(SearchResult(
                document=data["document"],
                score=combined_score,
                rank=0,  # Will be assigned after sorting
                method="hybrid"
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        # Return top_k results
        return combined_results[:top_k]


class RerankingRetrieval(RetrievalMethod):
    """
    Retrieval with reranking. Uses a base retriever to get initial results,
    then reranks them using a more sophisticated model.
    """
    
    def __init__(self, 
                base_retriever: RetrievalMethod,
                reranker: Any = None):
        """
        Initialize the reranking retrieval method.
        
        Args:
            base_retriever: Base retrieval method
            reranker: Reranking model or function
        """
        self.base_retriever = base_retriever
        self.reranker = reranker
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search with the base retriever and rerank results."""
        print(f"Reranking retrieval for query: {query}")
        
        # Get initial results from base retriever
        # Retrieve more results than needed for reranking
        initial_results = self.base_retriever.search(query, top_k=top_k * 3)
        
        # Rerank results
        reranked_results = self._rerank(query, initial_results)
        
        # Return top_k results
        return reranked_results[:top_k]
    
    def _rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results using a more sophisticated model."""
        # In a real implementation, this would use the reranker model
        # For demonstration, we use a simple heuristic
        
        reranked = []
        
        for result in results:
            # Calculate new score based on query term presence
            # This is a very simple reranking logic
            orig_score = result.score
            
            # Check query term presence
            doc_text = result.document.content.lower()
            query_terms = query.lower().split()
            
            term_matches = sum(1 for term in query_terms if term in doc_text)
            term_factor = term_matches / len(query_terms) if query_terms else 0
            
            # Combine original score with term presence
            new_score = 0.7 * orig_score + 0.3 * term_factor
            
            reranked.append(SearchResult(
                document=result.document,
                score=new_score,
                rank=0,  # Will be assigned after sorting
                method="reranked"
            ))
        
        # Sort by new score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1
        
        return reranked


def create_sample_documents(n: int = 20) -> List[Document]:
    """Create sample documents for testing."""
    documents = []
    
    topics = [
        ("Machine Learning", "Machine learning is a field of AI that enables systems to learn from data and improve without explicit programming."),
        ("Natural Language Processing", "NLP is a branch of AI that helps computers understand, interpret, and manipulate human language."),
        ("Vector Databases", "Vector databases store data as high-dimensional vectors, enabling efficient similarity search for applications like RAG."),
        ("Embedding Models", "Embedding models convert text or other data into numerical vectors that capture semantic meaning."),
        ("Retrieval Augmented Generation", "RAG combines retrieval methods with generative models to enhance output quality with external knowledge."),
    ]
    
    for i in range(n):
        topic_idx = i % len(topics)
        topic_name, topic_desc = topics[topic_idx]
        
        content = f"{topic_name}: {topic_desc} This is document {i+1} with more details about {topic_name.lower()}."
        
        # Create document
        doc = Document(
            id=f"doc_{i+1}",
            content=content,
            metadata={
                "topic": topic_name,
                "length": len(content),
                "index": i+1
            },
            # Random embedding for demonstration
            embedding=np.random.rand(768)
        )
        
        # Normalize embedding
        doc.embedding = doc.embedding / np.linalg.norm(doc.embedding)
        
        documents.append(doc)
    
    return documents


def compare_retrieval_methods(query: str, documents: List[Document], top_k: int = 5):
    """Compare different retrieval methods on the same query."""
    print(f"Comparing retrieval methods for query: '{query}'")
    
    # Create retrievers
    dense = DenseRetrieval(documents)
    sparse_bm25 = SparseRetrieval(documents, method="bm25")
    sparse_tfidf = SparseRetrieval(documents, method="tfidf")
    hybrid = HybridRetrieval(dense, sparse_bm25, alpha=0.5)
    reranking = RerankingRetrieval(hybrid)
    
    # Run searches
    dense_results = dense.search(query, top_k=top_k)
    sparse_bm25_results = sparse_bm25.search(query, top_k=top_k)
    sparse_tfidf_results = sparse_tfidf.search(query, top_k=top_k)
    hybrid_results = hybrid.search(query, top_k=top_k)
    reranked_results = reranking.search(query, top_k=top_k)
    
    # Print results
    methods = {
        "Dense": dense_results,
        "Sparse (BM25)": sparse_bm25_results,
        "Sparse (TF-IDF)": sparse_tfidf_results,
        "Hybrid": hybrid_results,
        "Reranked": reranked_results
    }
    
    for method_name, results in methods.items():
        print(f"\n{method_name} Retrieval Results:")
        print("-" * 40)
        
        for i, result in enumerate(results):
            print(f"{i+1}. [{result.score:.4f}] {result.document.content[:100]}...")
    
    return methods


if __name__ == "__main__":
    print("Hybrid Retrieval for RAG Systems")
    print("-------------------------------")
    
    # Create sample documents
    documents = create_sample_documents(25)
    print(f"Created {len(documents)} sample documents")
    
    # Test queries
    queries = [
        "What is machine learning?",
        "How do vector databases work?",
        "Explain retrieval augmented generation",
    ]
    
    # Compare methods for each query
    for query in queries:
        print("\n" + "=" * 80)
        results = compare_retrieval_methods(query, documents)
        print("=" * 80)
    
    print("\nConclusion:")
    print("Each retrieval method has strengths and weaknesses:")
    print("- Dense retrieval excels at semantic similarity but can miss keyword matches")
    print("- Sparse methods are great for exact terminology but miss semantic connections")
    print("- Hybrid approaches combine the strengths of both methods")
    print("- Reranking further improves precision by applying more sophisticated models")
    print("\nThe best approach depends on your specific use case and requirements.") 