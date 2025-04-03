"""
Reranking Methods for RAG Systems

This module provides implementations of various reranking strategies to improve 
the precision of retrieved documents for RAG applications.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import re
import time


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
    method: str = "default"  # The method that produced this result


class Reranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """
        Rerank search results based on the query.
        
        Args:
            query: The search query
            results: List of initial search results
            top_k: Number of top results to return
            
        Returns:
            Reranked list of search results
        """
        pass


class KeywordMatchReranker(Reranker):
    """
    Reranks results based on keyword matches between query and document content.
    Simple but effective for prioritizing documents with exact query terms.
    """
    
    def __init__(self, weight: float = 0.3):
        """
        Initialize the keyword match reranker.
        
        Args:
            weight: Weight to apply to keyword presence (relative to original score)
        """
        self.weight = weight
    
    def rerank(self, query: str, results: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """Rerank based on keyword matches."""
        print(f"Reranking with keyword matcher for query: {query}")
        
        # Tokenize query
        query_terms = self._tokenize(query)
        
        # Rerank results
        reranked = []
        
        for result in results:
            # Get original score
            orig_score = result.score
            
            # Calculate keyword match score
            doc_terms = self._tokenize(result.document.content)
            
            # Count matching terms
            matching_terms = set(query_terms).intersection(set(doc_terms))
            match_ratio = len(matching_terms) / len(query_terms) if query_terms else 0
            
            # Combine scores
            new_score = (1 - self.weight) * orig_score + self.weight * match_ratio
            
            reranked.append(SearchResult(
                document=result.document,
                score=new_score,
                rank=0,  # Will be assigned after sorting
                method="keyword_reranking"
            ))
        
        # Sort by new score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1
        
        # Return top_k results
        return reranked[:top_k]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization function."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Split by whitespace
        return text.split()


class CrossEncoderReranker(Reranker):
    """
    Uses a cross-encoder model to score query-document pairs.
    More accurate than other methods but computationally expensive.
    """
    
    def __init__(self, model: Optional[Any] = None):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model: Cross-encoder model for scoring query-document pairs
        """
        self.model = model
    
    def rerank(self, query: str, results: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """Rerank using cross-encoder model."""
        print(f"Reranking with cross-encoder for query: {query}")
        
        # In a real implementation, this would use the cross-encoder model
        # For demonstration, we simulate cross-encoder scores
        
        reranked = []
        
        for result in results:
            # Simulate cross-encoder score
            # In practice, this would pass query and document to the model
            orig_score = result.score
            
            # Generate a random factor with some bias towards original ranking
            # This is just for demonstration purposes
            random_factor = 0.7 + 0.3 * (1.0 - (result.rank / len(results)))
            
            # Simulate cross-encoder score
            # Mock implementation that favors documents containing query terms
            doc_text = result.document.content.lower()
            query_lower = query.lower()
            
            # Check if query terms are present in document
            cross_encoder_score = 0.5  # Base score
            
            # Boost score if query terms appear in document
            for term in query_lower.split():
                if term in doc_text:
                    cross_encoder_score += 0.1
            
            # Cap score at 1.0
            cross_encoder_score = min(cross_encoder_score * random_factor, 1.0)
            
            reranked.append(SearchResult(
                document=result.document,
                score=cross_encoder_score,
                rank=0,  # Will be assigned after sorting
                method="cross_encoder"
            ))
        
        # Sort by new score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1
        
        # Return top_k results
        return reranked[:top_k]


class RecencyReranker(Reranker):
    """
    Reranks results based on document recency.
    Useful for prioritizing more recent information.
    """
    
    def __init__(self, weight: float = 0.2, recency_field: str = "timestamp"):
        """
        Initialize the recency reranker.
        
        Args:
            weight: Weight to apply to recency score
            recency_field: Field in document metadata containing timestamp
        """
        self.weight = weight
        self.recency_field = recency_field
    
    def rerank(self, query: str, results: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """Rerank based on document recency."""
        print(f"Reranking with recency for query: {query}")
        
        # Check if documents have recency information
        timestamps = []
        for result in results:
            timestamp = result.document.metadata.get(self.recency_field)
            if timestamp is None:
                # If no timestamp, use document ID as a proxy (for demo)
                # In a real application, you'd want to handle this differently
                timestamp = int(result.document.id.split('_')[-1]) if result.document.id.split('_')[-1].isdigit() else 0
            timestamps.append(timestamp)
        
        # If no valid timestamps, return original results
        if not timestamps:
            return results[:top_k]
        
        # Find min and max for normalization
        min_timestamp = min(timestamps)
        max_timestamp = max(timestamps)
        timestamp_range = max_timestamp - min_timestamp
        
        # Rerank results
        reranked = []
        
        for i, result in enumerate(results):
            # Get original score
            orig_score = result.score
            
            # Calculate recency score
            timestamp = timestamps[i]
            
            # Normalize timestamp to [0, 1]
            recency_score = 0.0
            if timestamp_range > 0:
                recency_score = (timestamp - min_timestamp) / timestamp_range
            
            # Combine scores
            new_score = (1 - self.weight) * orig_score + self.weight * recency_score
            
            reranked.append(SearchResult(
                document=result.document,
                score=new_score,
                rank=0,  # Will be assigned after sorting
                method="recency_reranking"
            ))
        
        # Sort by new score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1
        
        # Return top_k results
        return reranked[:top_k]


class DiversityReranker(Reranker):
    """
    Reranks results to promote diversity in the top results.
    Useful for providing broader coverage of relevant information.
    """
    
    def __init__(self, 
                diversity_field: str = "topic", 
                similarity_threshold: float = 0.85):
        """
        Initialize the diversity reranker.
        
        Args:
            diversity_field: Field in document metadata to diversify on
            similarity_threshold: Threshold for considering documents similar
        """
        self.diversity_field = diversity_field
        self.similarity_threshold = similarity_threshold
    
    def rerank(self, query: str, results: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """Rerank to promote diversity."""
        print(f"Reranking for diversity for query: {query}")
        
        # If not enough results, return as is
        if len(results) <= top_k:
            return results
        
        # Check if documents have the diversity field
        has_field = False
        for result in results:
            if self.diversity_field in result.document.metadata:
                has_field = True
                break
        
        # If no diversity field, use content similarity
        if not has_field:
            return self._diversity_by_content(results, top_k)
        else:
            return self._diversity_by_field(results, top_k)
    
    def _diversity_by_field(self, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Diversify based on a metadata field."""
        selected = []
        remaining = list(results)
        
        # Always include the top result
        if remaining:
            selected.append(remaining.pop(0))
        
        # Add remaining results, prioritizing diversity
        while len(selected) < top_k and remaining:
            # Track diversity of current selection
            selected_categories = {}
            for result in selected:
                category = result.document.metadata.get(self.diversity_field, "unknown")
                if category in selected_categories:
                    selected_categories[category] += 1
                else:
                    selected_categories[category] = 1
            
            # Find the most diverse result
            best_result = None
            best_score = -1
            
            for i, result in enumerate(remaining):
                category = result.document.metadata.get(self.diversity_field, "unknown")
                
                # Calculate diversity score
                # Lower count of this category = higher diversity
                category_count = selected_categories.get(category, 0)
                diversity_score = 1.0 / (category_count + 1)
                
                # Combine with original relevance score
                combined_score = 0.7 * result.score + 0.3 * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_result = i
            
            # Add the best result
            if best_result is not None:
                selected.append(remaining.pop(best_result))
            else:
                # If no suitable result found, add the next one in order
                selected.append(remaining.pop(0))
        
        # Assign ranks
        for i, result in enumerate(selected):
            result.rank = i + 1
            result.method = "diversity_reranking"
        
        return selected
    
    def _diversity_by_content(self, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Diversify based on content similarity."""
        selected = []
        remaining = list(results)
        
        # Always include the top result
        if remaining:
            selected.append(remaining.pop(0))
        
        # Add remaining results, prioritizing diversity
        while len(selected) < top_k and remaining:
            # Find the most diverse result
            most_diverse_idx = 0
            most_diverse_score = -float('inf')
            
            for i, result in enumerate(remaining):
                # Calculate minimum similarity to any selected document
                min_similarity = float('inf')
                
                for sel in selected:
                    similarity = self._text_similarity(result.document.content, sel.document.content)
                    min_similarity = min(min_similarity, similarity)
                
                # If no selected documents yet, set a default
                if min_similarity == float('inf'):
                    min_similarity = 0.5
                
                # Calculate diversity score
                # Lower similarity = higher diversity
                diversity_score = 1.0 - min_similarity
                
                # Combine with original relevance score
                combined_score = 0.7 * result.score + 0.3 * diversity_score
                
                if combined_score > most_diverse_score:
                    most_diverse_score = combined_score
                    most_diverse_idx = i
            
            # Add the most diverse result
            selected.append(remaining.pop(most_diverse_idx))
        
        # Assign ranks
        for i, result in enumerate(selected):
            result.rank = i + 1
            result.method = "diversity_reranking"
        
        return selected
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        # Convert to sets of words
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization function."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Split by whitespace
        return text.split()


class EnsembleReranker(Reranker):
    """
    Combines multiple rerankers to get the benefits of each.
    Flexible approach that can be tuned for different use cases.
    """
    
    def __init__(self, rerankers: List[Tuple[Reranker, float]]):
        """
        Initialize the ensemble reranker.
        
        Args:
            rerankers: List of (reranker, weight) tuples
        """
        self.rerankers = rerankers
    
    def rerank(self, query: str, results: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """Rerank using an ensemble of rerankers."""
        print(f"Ensemble reranking for query: {query}")
        
        # If no rerankers, return original results
        if not self.rerankers:
            return results[:top_k]
        
        # Apply each reranker and collect scores
        all_scores = {}
        
        for reranker, weight in self.rerankers:
            # Apply reranker
            reranked = reranker.rerank(query, results)
            
            # Collect scores
            for result in reranked:
                doc_id = result.document.id
                if doc_id not in all_scores:
                    all_scores[doc_id] = {"document": result.document, "scores": []}
                
                # Add weighted score
                all_scores[doc_id]["scores"].append(weight * result.score)
        
        # Calculate combined scores
        combined_results = []
        
        for doc_id, data in all_scores.items():
            # Calculate average score
            avg_score = sum(data["scores"]) / sum(weight for _, weight in self.rerankers)
            
            combined_results.append(SearchResult(
                document=data["document"],
                score=avg_score,
                rank=0,  # Will be assigned after sorting
                method="ensemble_reranking"
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        # Return top_k results
        return combined_results[:top_k]


class CustomReranker(Reranker):
    """
    Custom reranker that uses a user-provided scoring function.
    Allows for domain-specific reranking logic.
    """
    
    def __init__(self, scoring_function: Callable[[str, Document], float]):
        """
        Initialize the custom reranker.
        
        Args:
            scoring_function: Function that takes query and document and returns a score
        """
        self.scoring_function = scoring_function
    
    def rerank(self, query: str, results: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """Rerank using custom scoring function."""
        print(f"Custom reranking for query: {query}")
        
        # Apply custom scoring function
        reranked = []
        
        for result in results:
            # Apply custom scoring
            new_score = self.scoring_function(query, result.document)
            
            reranked.append(SearchResult(
                document=result.document,
                score=new_score,
                rank=0,  # Will be assigned after sorting
                method="custom_reranking"
            ))
        
        # Sort by new score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1
        
        # Return top_k results
        return reranked[:top_k]


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
                "timestamp": 1600000000 + (i * 86400),  # Increasing timestamps
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


def create_sample_results(query: str, documents: List[Document], 
                         n: int = 10) -> List[SearchResult]:
    """Create sample search results for testing rerankers."""
    # Use subset of documents
    subset = documents[:min(n*2, len(documents))]
    
    # Assign random scores
    np.random.seed(42)  # For reproducibility
    
    results = []
    for i, doc in enumerate(subset):
        # Random score with some bias towards earlier documents
        score = 0.9 - (0.4 * i / len(subset)) + (0.1 * np.random.random())
        
        results.append(SearchResult(
            document=doc,
            score=score,
            rank=i+1,
            method="initial"
        ))
    
    return results[:n]


def compare_rerankers(query: str, results: List[SearchResult]):
    """Compare different rerankers on the same results."""
    print(f"Comparing rerankers for query: '{query}'")
    
    # Create rerankers
    keyword_reranker = KeywordMatchReranker(weight=0.3)
    cross_encoder_reranker = CrossEncoderReranker()
    recency_reranker = RecencyReranker(weight=0.3)
    diversity_reranker = DiversityReranker()
    
    # Create ensemble reranker
    ensemble_reranker = EnsembleReranker([
        (keyword_reranker, 0.3),
        (cross_encoder_reranker, 0.5),
        (recency_reranker, 0.2)
    ])
    
    # Define custom scoring function
    def custom_scoring(query: str, document: Document) -> float:
        # Example: score based on document length and query term presence
        base_score = 0.5
        
        # Prefer shorter documents
        length = len(document.content)
        length_factor = 1.0 - min(length / 1000, 0.5)  # Penalize long documents
        
        # Check query term presence
        query_terms = query.lower().split()
        content_lower = document.content.lower()
        
        term_presence = sum(1 for term in query_terms if term in content_lower)
        term_factor = term_presence / len(query_terms) if query_terms else 0
        
        # Combine factors
        return 0.2 * base_score + 0.3 * length_factor + 0.5 * term_factor
    
    custom_reranker = CustomReranker(scoring_function=custom_scoring)
    
    # Apply rerankers
    keyword_results = keyword_reranker.rerank(query, results)
    cross_encoder_results = cross_encoder_reranker.rerank(query, results)
    recency_results = recency_reranker.rerank(query, results)
    diversity_results = diversity_reranker.rerank(query, results)
    ensemble_results = ensemble_reranker.rerank(query, results)
    custom_results = custom_reranker.rerank(query, results)
    
    # Print results
    reranker_results = {
        "Original": results,
        "Keyword Match": keyword_results,
        "Cross-Encoder": cross_encoder_results,
        "Recency": recency_results,
        "Diversity": diversity_results,
        "Ensemble": ensemble_results,
        "Custom": custom_results
    }
    
    for name, res in reranker_results.items():
        print(f"\n{name} Reranker Results:")
        print("-" * 40)
        
        for i, result in enumerate(res[:5]):  # Show top 5
            doc = result.document
            print(f"{i+1}. [{result.score:.4f}] Topic: {doc.metadata.get('topic', 'Unknown')}")
            print(f"   {doc.content[:80]}...")
    
    return reranker_results


if __name__ == "__main__":
    print("Reranking Methods for RAG Systems")
    print("--------------------------------")
    
    # Create sample documents
    documents = create_sample_documents(25)
    print(f"Created {len(documents)} sample documents")
    
    # Test queries
    queries = [
        "What is machine learning?",
        "How do vector databases work?",
        "Explain retrieval augmented generation",
    ]
    
    # Compare rerankers for each query
    for query in queries:
        print("\n" + "=" * 80)
        initial_results = create_sample_results(query, documents, 10)
        reranker_results = compare_rerankers(query, initial_results)
        print("=" * 80)
    
    print("\nConclusion:")
    print("Each reranking method optimizes for different aspects:")
    print("- Keyword matching prioritizes documents containing query terms")
    print("- Cross-encoders provide more accurate relevance assessment")
    print("- Recency prioritizes newer documents")
    print("- Diversity ensures coverage of different topics")
    print("- Ensemble combines multiple approaches")
    print("- Custom allows for domain-specific logic")
    print("\nChoose the reranker(s) that best align with your application's requirements.") 