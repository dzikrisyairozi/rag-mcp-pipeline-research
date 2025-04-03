"""
Context Window Optimization for RAG Systems

This module provides implementations of various optimization techniques for managing
context windows in RAG applications, including prioritization, compression, 
and token management strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import re
import time


@dataclass
class Document:
    """Represents a document in the retrieval system."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class TokenCount:
    """Information about token counts for a piece of text."""
    text: str
    token_count: int
    char_count: int


class ContextWindowOptimizer(ABC):
    """Abstract base class for context window optimizers."""
    
    @abstractmethod
    def optimize(self, documents: List[Document], 
                max_tokens: int, query: Optional[str] = None) -> str:
        """
        Optimize a list of documents to fit within a token limit.
        
        Args:
            documents: List of documents to optimize
            max_tokens: Maximum allowed tokens
            query: Optional query for relevance-based optimization
            
        Returns:
            Optimized context as a string
        """
        pass
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate the number of tokens in a piece of text.
        This is a very rough approximation. In practice, use a proper tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Very rough approximation: ~4 chars per token for English text
        return len(text) // 4


class PrioritizationOptimizer(ContextWindowOptimizer):
    """
    Optimizes context window by prioritizing documents based on relevance scores.
    Simple but effective approach that selects the most relevant content.
    """
    
    def __init__(self, overlap_tokens: int = 0):
        """
        Initialize the optimizer.
        
        Args:
            overlap_tokens: Number of tokens to allow for overlap between documents
        """
        self.overlap_tokens = overlap_tokens
    
    def optimize(self, documents: List[Document], 
                max_tokens: int, query: Optional[str] = None) -> str:
        """
        Optimize by prioritizing documents based on relevance score.
        
        Args:
            documents: List of documents to optimize
            max_tokens: Maximum allowed tokens
            query: Optional query (not used in this optimizer)
            
        Returns:
            Optimized context as a string
        """
        # Sort documents by score (descending)
        sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
        
        # Build context until we hit the token limit
        context_parts = []
        total_tokens = 0
        
        for doc in sorted_docs:
            # Estimate tokens in this document
            doc_tokens = self.estimate_tokens(doc.content)
            
            # Check if adding this document would exceed the limit
            if total_tokens + doc_tokens > max_tokens:
                # If we're close to the limit, try to add a partial document
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 50:  # Only add if we have a meaningful amount of space left
                    partial_content = self._truncate_to_tokens(doc.content, remaining_tokens)
                    context_parts.append(partial_content)
                break
            
            # Add document
            context_parts.append(doc.content)
            total_tokens += doc_tokens
            
            # Subtract overlap for next document
            total_tokens -= self.overlap_tokens
        
        # Join context parts
        return "\n\n".join(context_parts)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to a maximum number of tokens.
        Tries to break at sentence boundaries when possible.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens to keep
            
        Returns:
            Truncated text
        """
        # Estimate characters based on token count
        estimated_chars = max_tokens * 4
        
        # If text is already within limit, return as is
        if len(text) <= estimated_chars:
            return text
        
        # Try to find a good sentence boundary
        truncated_text = text[:estimated_chars]
        last_period = truncated_text.rfind('. ')
        
        if last_period > estimated_chars * 0.5:
            truncated_text = text[:last_period + 1]
        
        return truncated_text


class RelevanceOptimizer(ContextWindowOptimizer):
    """
    Optimizes context window by recalculating relevance to the query.
    More sophisticated than simple prioritization, taking query into account.
    """
    
    def __init__(self, relevance_function: Optional[Callable[[str, str], float]] = None):
        """
        Initialize the optimizer.
        
        Args:
            relevance_function: Optional function to calculate relevance between query and text
        """
        self.relevance_function = relevance_function
    
    def optimize(self, documents: List[Document], 
                max_tokens: int, query: Optional[str] = None) -> str:
        """
        Optimize by recalculating relevance to the query.
        
        Args:
            documents: List of documents to optimize
            max_tokens: Maximum allowed tokens
            query: Query for relevance-based optimization
            
        Returns:
            Optimized context as a string
        """
        # If no query provided, fall back to prioritization
        if not query:
            optimizer = PrioritizationOptimizer()
            return optimizer.optimize(documents, max_tokens)
        
        # Recalculate relevance for each document
        docs_with_relevance = []
        
        for doc in documents:
            if self.relevance_function:
                # Use provided relevance function
                relevance = self.relevance_function(query, doc.content)
            else:
                # Simple fallback: count query terms in document
                relevance = self._simple_relevance(query, doc.content)
            
            docs_with_relevance.append((doc, relevance))
        
        # Sort documents by relevance (descending)
        sorted_docs = [item[0] for item in sorted(docs_with_relevance, key=lambda x: x[1], reverse=True)]
        
        # Build context until we hit the token limit
        context_parts = []
        total_tokens = 0
        
        for doc in sorted_docs:
            # Estimate tokens in this document
            doc_tokens = self.estimate_tokens(doc.content)
            
            # Check if adding this document would exceed the limit
            if total_tokens + doc_tokens > max_tokens:
                # If we're close to the limit, try to add a partial document
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 50:  # Only add if we have a meaningful amount of space left
                    partial_content = self._get_most_relevant_part(doc.content, query, remaining_tokens)
                    context_parts.append(partial_content)
                break
            
            # Add document
            context_parts.append(doc.content)
            total_tokens += doc_tokens
        
        # Join context parts
        return "\n\n".join(context_parts)
    
    def _simple_relevance(self, query: str, text: str) -> float:
        """
        Calculate simple relevance score based on query term presence.
        
        Args:
            query: Query string
            text: Document text
            
        Returns:
            Relevance score
        """
        query_terms = re.findall(r'\w+', query.lower())
        text_lower = text.lower()
        
        # Count occurrences of query terms
        total_occurrences = 0
        
        for term in query_terms:
            if len(term) < 3:  # Skip very short terms
                continue
            
            occurrences = text_lower.count(term)
            total_occurrences += occurrences
        
        # Normalize by document length
        return total_occurrences / (len(text) / 100) if text else 0
    
    def _get_most_relevant_part(self, text: str, query: str, max_tokens: int) -> str:
        """
        Extract the most relevant part of the text for the query.
        
        Args:
            text: Text to extract from
            query: Query string
            max_tokens: Maximum tokens to extract
            
        Returns:
            Most relevant part of the text
        """
        estimated_chars = max_tokens * 4
        
        # If text is already within limit, return as is
        if len(text) <= estimated_chars:
            return text
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Calculate relevance of each sentence
        sentence_relevance = []
        
        for sentence in sentences:
            relevance = self._simple_relevance(query, sentence)
            sentence_relevance.append((sentence, relevance))
        
        # Sort sentences by relevance
        sorted_sentences = sorted(sentence_relevance, key=lambda x: x[1], reverse=True)
        
        # Take most relevant sentences until we hit the token limit
        selected_sentences = []
        total_chars = 0
        
        for sentence, _ in sorted_sentences:
            if total_chars + len(sentence) + 1 > estimated_chars:  # +1 for space
                break
            
            selected_sentences.append(sentence)
            total_chars += len(sentence) + 1  # +1 for space
        
        # Join selected sentences
        return " ".join(selected_sentences)


class CompressionOptimizer(ContextWindowOptimizer):
    """
    Optimizes context window by compressing document content.
    Uses techniques like removing redundancies and summarization.
    """
    
    def __init__(self, summarizer: Optional[Callable[[str, int], str]] = None):
        """
        Initialize the optimizer.
        
        Args:
            summarizer: Optional function to summarize text
        """
        self.summarizer = summarizer
    
    def optimize(self, documents: List[Document], 
                max_tokens: int, query: Optional[str] = None) -> str:
        """
        Optimize by compressing document content.
        
        Args:
            documents: List of documents to optimize
            max_tokens: Maximum allowed tokens
            query: Optional query for relevance-based compression
            
        Returns:
            Optimized context as a string
        """
        # Sort documents by score (descending)
        sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
        
        # Calculate total tokens in all documents
        total_doc_tokens = sum(self.estimate_tokens(doc.content) for doc in sorted_docs)
        
        # If total tokens is within limit, no compression needed
        if total_doc_tokens <= max_tokens:
            return "\n\n".join(doc.content for doc in sorted_docs)
        
        # Calculate compression ratio
        compression_ratio = max_tokens / total_doc_tokens
        
        # Compress each document
        compressed_parts = []
        
        for doc in sorted_docs:
            # Calculate target tokens for this document
            doc_tokens = self.estimate_tokens(doc.content)
            target_tokens = int(doc_tokens * compression_ratio)
            
            # Skip very small allocations
            if target_tokens < 10:
                continue
            
            # Compress document
            compressed_content = self._compress_text(doc.content, target_tokens)
            compressed_parts.append(compressed_content)
        
        # Join compressed parts
        return "\n\n".join(compressed_parts)
    
    def _compress_text(self, text: str, target_tokens: int) -> str:
        """
        Compress text to target token count.
        
        Args:
            text: Text to compress
            target_tokens: Target token count
            
        Returns:
            Compressed text
        """
        # If summarizer provided, use it
        if self.summarizer:
            return self.summarizer(text, target_tokens)
        
        # Simple fallback: truncate and add indicator
        estimated_chars = target_tokens * 4
        
        if len(text) <= estimated_chars:
            return text
        
        # Try to find a good sentence boundary
        truncated = text[:estimated_chars]
        last_period = truncated.rfind('. ')
        
        if last_period > estimated_chars * 0.5:
            return text[:last_period + 1] + " [...]"
        else:
            return truncated + " [...]"


class TokenManagementOptimizer(ContextWindowOptimizer):
    """
    Optimizes context window by allocating tokens among sections.
    Sophisticated approach that balances different context components.
    """
    
    def __init__(self, sections: List[Tuple[str, float]] = None):
        """
        Initialize the optimizer.
        
        Args:
            sections: List of (section_name, weight) tuples
        """
        self.sections = sections or [
            ("query", 0.05),       # User query
            ("documents", 0.75),   # Retrieved documents
            ("examples", 0.15),    # Few-shot examples
            ("instructions", 0.05) # System instructions
        ]
    
    def optimize(self, documents: List[Document], 
                max_tokens: int, query: Optional[str] = None) -> str:
        """
        Optimize by allocating tokens among sections.
        
        Args:
            documents: List of documents to optimize
            max_tokens: Maximum allowed tokens
            query: Optional query to include
            
        Returns:
            Optimized context as a string
        """
        # Create section allocations
        allocations = {}
        remaining_tokens = max_tokens
        
        for section_name, weight in self.sections:
            # Calculate token allocation for this section
            allocation = int(max_tokens * weight)
            allocations[section_name] = allocation
            remaining_tokens -= allocation
        
        # If we have remaining tokens, add them to documents section
        if remaining_tokens > 0:
            allocations["documents"] += remaining_tokens
        
        # Process documents with prioritization
        documents_optimizer = PrioritizationOptimizer()
        doc_context = documents_optimizer.optimize(documents, allocations["documents"])
        
        # Build final context
        context_parts = []
        
        # Add query if provided
        if query and allocations.get("query", 0) > 0:
            query_tokens = self.estimate_tokens(query)
            if query_tokens > allocations["query"]:
                # Truncate query if too long
                query = query[:allocations["query"] * 4]
            context_parts.append(f"Query: {query}")
        
        # Add documents
        context_parts.append(f"Documents:\n{doc_context}")
        
        # In a real implementation, you would add examples and instructions
        # based on their allocations. Here we just include placeholders.
        if allocations.get("examples", 0) > 0:
            context_parts.append("Examples: [Few-shot examples would go here]")
        
        if allocations.get("instructions", 0) > 0:
            context_parts.append("Instructions: [System instructions would go here]")
        
        # Join context parts
        return "\n\n".join(context_parts)


class DynamicContextManager:
    """
    Manages context window dynamically based on conversation history and query.
    Advanced approach for multi-turn conversations.
    """
    
    def __init__(self, max_tokens: int, history_weight: float = 0.2):
        """
        Initialize the context manager.
        
        Args:
            max_tokens: Maximum context window size in tokens
            history_weight: Weight for conversation history
        """
        self.max_tokens = max_tokens
        self.history_weight = history_weight
        self.conversation_history: List[Dict[str, str]] = []
        self.optimizer = TokenManagementOptimizer([
            ("history", history_weight),
            ("query", 0.05),
            ("documents", 0.75)
        ])
    
    def add_exchange(self, query: str, response: str):
        """
        Add a conversation exchange to history.
        
        Args:
            query: User query
            response: System response
        """
        self.conversation_history.append({
            "query": query,
            "response": response,
            "timestamp": time.time()
        })
    
    def get_context(self, query: str, documents: List[Document]) -> str:
        """
        Get optimized context for the current query.
        
        Args:
            query: Current user query
            documents: Retrieved documents
            
        Returns:
            Optimized context as a string
        """
        # Calculate token allocation
        history_tokens = int(self.max_tokens * self.history_weight)
        remaining_tokens = self.max_tokens - history_tokens
        
        # Format conversation history
        history_context = self._format_history(history_tokens)
        
        # Optimize documents
        documents_optimizer = PrioritizationOptimizer()
        doc_context = documents_optimizer.optimize(documents, remaining_tokens - self.estimate_tokens(query), query)
        
        # Combine contexts
        if history_context:
            return f"Conversation History:\n{history_context}\n\nCurrent Query: {query}\n\nRetrieved Documents:\n{doc_context}"
        else:
            return f"Query: {query}\n\nRetrieved Documents:\n{doc_context}"
    
    def _format_history(self, max_tokens: int) -> str:
        """
        Format conversation history to fit within token limit.
        
        Args:
            max_tokens: Maximum tokens for history
            
        Returns:
            Formatted history as a string
        """
        if not self.conversation_history:
            return ""
        
        # Start with most recent exchanges
        history = list(reversed(self.conversation_history))
        formatted_exchanges = []
        total_tokens = 0
        
        for exchange in history:
            # Format exchange
            formatted = f"User: {exchange['query']}\nSystem: {exchange['response']}"
            exchange_tokens = self.estimate_tokens(formatted)
            
            # Check if adding this exchange would exceed the limit
            if total_tokens + exchange_tokens > max_tokens:
                # If we're close to the limit, try to add a partial exchange
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 50:  # Only add if we have a meaningful amount of space left
                    truncated = formatted[:remaining_tokens * 4]
                    formatted_exchanges.append(truncated + " [...]")
                break
            
            # Add exchange
            formatted_exchanges.append(formatted)
            total_tokens += exchange_tokens
        
        # Reverse back to chronological order
        formatted_exchanges.reverse()
        
        # Join exchanges
        return "\n\n".join(formatted_exchanges)


def create_sample_documents(n: int = 5) -> List[Document]:
    """Create sample documents for testing optimizers."""
    documents = []
    
    topics = [
        ("Vector Databases", "Vector databases store data as high-dimensional vectors, enabling efficient similarity search for applications like RAG. They index vectors using techniques such as approximate nearest neighbor search to ensure quick retrieval even with large datasets. Popular vector databases include Pinecone, Weaviate, Chroma, Qdrant, and FAISS."),
        ("Embedding Models", "Embedding models convert text or other data into numerical vectors that capture semantic meaning. These dense vector representations place similar items close together in the vector space, enabling semantic search beyond simple keyword matching. Common embedding models include OpenAI's text-embedding-ada-002, sentence-transformers models like mpnet, and E5 models."),
        ("Chunking Strategies", "Effective chunking strategies are crucial for RAG performance. Documents can be split by fixed size, semantic boundaries, or recursive approaches. The ideal chunking approach balances granularity (enabling precise retrieval) with context preservation (maintaining sufficient information in each chunk). Overlapping chunks can help preserve context across chunk boundaries."),
        ("Retrieval Methods", "RAG systems can use various retrieval methods, from simple BM25 keyword search to dense vector retrieval and hybrid approaches. Hybrid retrieval combines the strengths of sparse and dense methods, while reranking techniques can further improve precision by applying more sophisticated models to an initial set of retrieved documents."),
        ("Context Window Management", "Managing context windows effectively involves balancing different components like conversation history, retrieved documents, and system instructions. Techniques include token allocation, compression, prioritization, and dynamic adjustment based on query complexity. Proper context management ensures the LLM has the most relevant information within token limits."),
    ]
    
    for i in range(min(n, len(topics))):
        topic_name, topic_desc = topics[i]
        
        content = f"{topic_name}: {topic_desc}"
        
        # Create document with descending scores
        doc = Document(
            id=f"doc_{i+1}",
            content=content,
            metadata={
                "topic": topic_name,
                "length": len(content)
            },
            score=0.9 - (i * 0.1)  # Descending scores
        )
        
        documents.append(doc)
    
    return documents


def demonstrate_optimizers():
    """Demonstrate different context window optimizers."""
    print("Context Window Optimization for RAG Systems")
    print("-----------------------------------------")
    
    # Create sample documents
    documents = create_sample_documents(5)
    print(f"Created {len(documents)} sample documents")
    
    # Example query
    query = "How do I manage context windows in RAG applications?"
    
    # Create optimizers
    optimizers = {
        "Prioritization": PrioritizationOptimizer(),
        "Relevance": RelevanceOptimizer(),
        "Compression": CompressionOptimizer(),
        "Token Management": TokenManagementOptimizer()
    }
    
    # Test each optimizer
    for name, optimizer in optimizers.items():
        print("\n" + "=" * 80)
        print(f"Optimizer: {name}")
        print("-" * 80)
        
        # Optimize context
        max_tokens = 150  # Small limit for demonstration
        context = optimizer.optimize(documents, max_tokens, query)
        
        # Calculate token count
        token_count = optimizer.estimate_tokens(context)
        
        print(f"Token count: ~{token_count} / {max_tokens}")
        print("\nOptimized Context:")
        print(context[:500] + ("..." if len(context) > 500 else ""))
        
        print("=" * 80)
    
    # Demonstrate dynamic context manager
    print("\n" + "=" * 80)
    print("Dynamic Context Manager")
    print("-" * 80)
    
    manager = DynamicContextManager(max_tokens=200)
    
    # Add some conversation history
    manager.add_exchange(
        "What are vector databases?",
        "Vector databases store embeddings for similarity search, with examples including Pinecone and Chroma."
    )
    manager.add_exchange(
        "How are they used in RAG?",
        "In RAG systems, vector databases store document embeddings and enable quick retrieval of relevant context."
    )
    
    # Get context for current query
    context = manager.get_context(query, documents)
    
    # Calculate token count
    token_count = optimizer.estimate_tokens(context)
    
    print(f"Token count: ~{token_count} / 200")
    print("\nDynamic Context:")
    print(context[:500] + ("..." if len(context) > 500 else ""))
    
    print("=" * 80)
    
    print("\nConclusion:")
    print("Each context optimization strategy has different strengths:")
    print("- Prioritization is simple but effective when document scores are reliable")
    print("- Relevance optimization improves results by considering the query")
    print("- Compression helps fit more information in the context window")
    print("- Token management balances different context components")
    print("- Dynamic management handles evolving conversations")
    print("\nThe best approach depends on your specific RAG application and requirements.")


if __name__ == "__main__":
    demonstrate_optimizers() 