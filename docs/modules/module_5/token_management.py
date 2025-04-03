"""
Token Management for RAG Systems

This module provides specialized techniques for managing and optimizing token usage 
in RAG (Retrieval Augmented Generation) systems, helping to maximize context utilization
within token limits of language models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import re
import time
import json


@dataclass
class Document:
    """Represents a document in the RAG system."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class TokenBudget:
    """Represents a token budget allocation for different components."""
    total: int
    allocations: Dict[str, int] = field(default_factory=dict)
    used: Dict[str, int] = field(default_factory=dict)
    
    def remaining(self, component: Optional[str] = None) -> int:
        """
        Get remaining tokens overall or for a specific component.
        
        Args:
            component: Optional component name
            
        Returns:
            Remaining tokens
        """
        if component:
            allocation = self.allocations.get(component, 0)
            used = self.used.get(component, 0)
            return max(0, allocation - used)
        else:
            used_total = sum(self.used.values())
            return max(0, self.total - used_total)
    
    def use(self, component: str, tokens: int) -> bool:
        """
        Record token usage for a component.
        
        Args:
            component: Component name
            tokens: Number of tokens used
            
        Returns:
            True if tokens were within budget, False otherwise
        """
        if component not in self.allocations:
            return False
        
        available = self.remaining(component)
        
        if tokens > available:
            # Cannot use more tokens than available
            return False
        
        # Record usage
        if component in self.used:
            self.used[component] += tokens
        else:
            self.used[component] = tokens
        
        return True


class TokenEstimator:
    """Estimates token counts for text."""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate tokens in text.
        This is a rough approximation. In practice, use a model-specific tokenizer.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 chars per token for English text
        return len(text) // 4
    
    @staticmethod
    def estimate_tokens_by_model(text: str, model: str) -> int:
        """
        Estimate tokens based on specific model characteristics.
        
        Args:
            text: Text to estimate tokens for
            model: Model name
            
        Returns:
            Estimated token count
        """
        # Model-specific estimates
        if model.startswith("gpt-4"):
            # GPT-4 tokenizer (approximately)
            return TokenEstimator.estimate_tokens(text)
        elif model.startswith("gpt-3.5"):
            # GPT-3.5 tokenizer (approximately)
            return TokenEstimator.estimate_tokens(text)
        elif model.startswith("claude"):
            # Claude tokenizer (approximately)
            return len(text) // 3.5  # Claude tends to have slightly smaller tokens
        else:
            # Default approximation
            return TokenEstimator.estimate_tokens(text)


class TokenAllocationStrategy(ABC):
    """Abstract base class for token allocation strategies."""
    
    @abstractmethod
    def allocate(self, total_tokens: int, components: List[str]) -> Dict[str, int]:
        """
        Allocate tokens among components.
        
        Args:
            total_tokens: Total tokens available
            components: List of component names
            
        Returns:
            Dictionary mapping component names to token allocations
        """
        pass


class FixedRatioAllocation(TokenAllocationStrategy):
    """Allocates tokens based on fixed ratios."""
    
    def __init__(self, ratios: Dict[str, float]):
        """
        Initialize with fixed ratios.
        
        Args:
            ratios: Dictionary mapping component names to ratios (should sum to 1.0)
        """
        self.ratios = ratios
        
        # Normalize ratios if they don't sum to 1.0
        total = sum(ratios.values())
        if abs(total - 1.0) > 0.001:  # Allow small floating point error
            self.ratios = {k: v / total for k, v in ratios.items()}
    
    def allocate(self, total_tokens: int, components: List[str]) -> Dict[str, int]:
        """
        Allocate tokens based on fixed ratios.
        
        Args:
            total_tokens: Total tokens available
            components: List of component names
            
        Returns:
            Dictionary mapping component names to token allocations
        """
        allocations = {}
        remaining = total_tokens
        
        # First pass: allocate based on ratios
        for component in components:
            if component in self.ratios:
                allocation = int(total_tokens * self.ratios[component])
                allocations[component] = allocation
                remaining -= allocation
            else:
                allocations[component] = 0
        
        # Second pass: distribute any remaining tokens to avoid wastage
        if remaining > 0:
            # Give to the component with the highest ratio
            max_ratio_component = max(self.ratios.items(), key=lambda x: x[1])[0]
            if max_ratio_component in allocations:
                allocations[max_ratio_component] += remaining
        
        return allocations


class DynamicAllocation(TokenAllocationStrategy):
    """Allocates tokens dynamically based on content importance."""
    
    def __init__(self, base_allocation: Dict[str, float], 
                 importance_fn: Optional[Callable[[str, Any], float]] = None):
        """
        Initialize with base allocation and importance function.
        
        Args:
            base_allocation: Base allocation ratios
            importance_fn: Function to determine component importance
        """
        self.base_allocation = base_allocation
        self.importance_fn = importance_fn
    
    def allocate(self, total_tokens: int, components: List[str], 
                content: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """
        Allocate tokens dynamically based on content importance.
        
        Args:
            total_tokens: Total tokens available
            components: List of component names
            content: Optional content to evaluate importance
            
        Returns:
            Dictionary mapping component names to token allocations
        """
        # Start with base allocation
        allocations = {}
        importance_scores = {}
        remaining = total_tokens
        
        # First pass: calculate importance and base allocations
        for component in components:
            if component in self.base_allocation:
                base_allocation = int(total_tokens * self.base_allocation[component])
                allocations[component] = base_allocation
                remaining -= base_allocation
                
                # Calculate importance if content provided
                if content and self.importance_fn and component in content:
                    importance = self.importance_fn(component, content[component])
                    importance_scores[component] = importance
        
        # Second pass: adjust based on importance
        if importance_scores and remaining > 0:
            # Normalize importance scores
            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                normalized_scores = {k: v / total_importance for k, v in importance_scores.items()}
                
                # Distribute remaining tokens by importance
                for component, score in normalized_scores.items():
                    additional = int(remaining * score)
                    allocations[component] += additional
                    remaining -= additional
        
        # Final pass: assign any remaining tokens
        if remaining > 0:
            # Distribute evenly among components
            components_with_allocation = [c for c in components if c in allocations]
            if components_with_allocation:
                per_component = remaining // len(components_with_allocation)
                for component in components_with_allocation:
                    allocations[component] += per_component
                    remaining -= per_component
                
                # Give any final remainder to first component
                if remaining > 0 and components_with_allocation:
                    allocations[components_with_allocation[0]] += remaining
        
        return allocations


class AdaptiveTokenManager:
    """
    Manages token allocation adaptively based on content and context.
    Provides dynamic adjustment between components like query, documents, history, etc.
    """
    
    def __init__(self, 
                total_tokens: int,
                allocation_strategy: TokenAllocationStrategy,
                estimator: TokenEstimator = TokenEstimator()):
        """
        Initialize the manager.
        
        Args:
            total_tokens: Total token budget
            allocation_strategy: Strategy for allocating tokens
            estimator: Token estimator
        """
        self.total_tokens = total_tokens
        self.allocation_strategy = allocation_strategy
        self.estimator = estimator
        self.token_budget = None
        self.components = set()
    
    def setup_budget(self, components: List[str], 
                    content: Optional[Dict[str, Any]] = None) -> TokenBudget:
        """
        Set up token budget for components.
        
        Args:
            components: List of component names
            content: Optional content information for dynamic allocation
            
        Returns:
            Token budget
        """
        # Allocate tokens
        if hasattr(self.allocation_strategy, 'allocate') and \
           callable(getattr(self.allocation_strategy, 'allocate')) and \
           'content' in self.allocation_strategy.allocate.__code__.co_varnames:
            # Strategy supports content-based allocation
            allocations = self.allocation_strategy.allocate(
                self.total_tokens, components, content
            )
        else:
            # Strategy doesn't need content
            allocations = self.allocation_strategy.allocate(
                self.total_tokens, components
            )
        
        # Create budget
        self.token_budget = TokenBudget(
            total=self.total_tokens,
            allocations=allocations,
            used={component: 0 for component in components}
        )
        
        # Store components
        self.components = set(components)
        
        return self.token_budget
    
    def fit_content(self, component: str, content: str, 
                   preserve_meaning: bool = True) -> str:
        """
        Fit content to allocated tokens.
        
        Args:
            component: Component name
            content: Content to fit
            preserve_meaning: Whether to try to preserve meaning
            
        Returns:
            Fitted content
        """
        if not self.token_budget or component not in self.token_budget.allocations:
            return content
        
        # Get allocation
        allocation = self.token_budget.allocations[component]
        
        # Estimate current tokens
        estimated_tokens = self.estimator.estimate_tokens(content)
        
        # If already fits, return as is
        if estimated_tokens <= allocation:
            # Record usage
            self.token_budget.use(component, estimated_tokens)
            return content
        
        # Need to truncate
        if preserve_meaning:
            # Try to preserve meaning by truncating at sentence boundaries
            return self._truncate_preserving_meaning(content, allocation)
        else:
            # Simple truncation
            chars_estimate = allocation * 4  # Rough estimate of chars per token
            truncated = content[:chars_estimate]
            
            # Record usage
            self.token_budget.use(component, allocation)
            
            return truncated
    
    def _truncate_preserving_meaning(self, content: str, token_limit: int) -> str:
        """
        Truncate content while trying to preserve meaning.
        
        Args:
            content: Content to truncate
            token_limit: Token limit
            
        Returns:
            Truncated content
        """
        # Rough estimate of chars
        char_limit = token_limit * 4
        
        # If content is already short enough, return as is
        if len(content) <= char_limit:
            self.token_budget.use(component, self.estimator.estimate_tokens(content))
            return content
        
        # Try to find a sentence boundary
        truncated = content[:char_limit]
        last_period = truncated.rfind('. ')
        
        if last_period > char_limit * 0.5:  # Only use if at least halfway through
            truncated = content[:last_period + 1]
        
        # Estimate truncated tokens
        truncated_tokens = self.estimator.estimate_tokens(truncated)
        
        # Record usage
        self.token_budget.use(component, truncated_tokens)
        
        return truncated
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Returns:
            Dictionary with token statistics
        """
        if not self.token_budget:
            return {}
        
        stats = {
            "total_budget": self.token_budget.total,
            "total_used": sum(self.token_budget.used.values()),
            "total_remaining": self.token_budget.remaining(),
            "components": {}
        }
        
        for component in self.components:
            if component in self.token_budget.allocations:
                stats["components"][component] = {
                    "allocation": self.token_budget.allocations.get(component, 0),
                    "used": self.token_budget.used.get(component, 0),
                    "remaining": self.token_budget.remaining(component)
                }
        
        return stats


class TokenBufferManager:
    """
    Manages token buffers for efficient context window utilization.
    Provides optimized token usage across multiple components like system prompt, 
    conversation history, and retrieved documents.
    """
    
    def __init__(self, 
                total_tokens: int,
                reserved_tokens: Dict[str, int] = None,
                buffer_tokens: int = 50):
        """
        Initialize the manager.
        
        Args:
            total_tokens: Total token budget
            reserved_tokens: Tokens to reserve for specific purposes
            buffer_tokens: Token buffer for safety
        """
        self.total_tokens = total_tokens
        self.reserved_tokens = reserved_tokens or {}
        self.buffer_tokens = buffer_tokens
        self.available_tokens = total_tokens - sum(self.reserved_tokens.values()) - buffer_tokens
        self.used_tokens = {}
    
    def allocate(self, component: str, tokens: int) -> bool:
        """
        Allocate tokens to a component.
        
        Args:
            component: Component name
            tokens: Number of tokens to allocate
            
        Returns:
            True if allocation successful, False otherwise
        """
        # Check if component has reserved tokens
        if component in self.reserved_tokens:
            # Cannot exceed reserved tokens
            if tokens > self.reserved_tokens[component]:
                return False
        else:
            # Check if enough tokens available
            if tokens > self.available_tokens:
                return False
            
            # Reduce available tokens
            self.available_tokens -= tokens
        
        # Record allocation
        if component in self.used_tokens:
            self.used_tokens[component] += tokens
        else:
            self.used_tokens[component] = tokens
        
        return True
    
    def release(self, component: str, tokens: int = None) -> int:
        """
        Release tokens from a component.
        
        Args:
            component: Component name
            tokens: Number of tokens to release (None for all)
            
        Returns:
            Number of tokens released
        """
        if component not in self.used_tokens:
            return 0
        
        if tokens is None:
            # Release all
            tokens_to_release = self.used_tokens[component]
            self.used_tokens[component] = 0
        else:
            # Release specific amount
            tokens_to_release = min(tokens, self.used_tokens[component])
            self.used_tokens[component] -= tokens_to_release
        
        # Check if component has reserved tokens
        if component not in self.reserved_tokens:
            # Return to available pool
            self.available_tokens += tokens_to_release
        
        return tokens_to_release
    
    def get_available(self) -> int:
        """
        Get available tokens.
        
        Returns:
            Number of available tokens
        """
        return self.available_tokens
    
    def get_used(self, component: str = None) -> int:
        """
        Get used tokens for a component or total.
        
        Args:
            component: Optional component name
            
        Returns:
            Number of used tokens
        """
        if component:
            return self.used_tokens.get(component, 0)
        else:
            return sum(self.used_tokens.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Returns:
            Dictionary with token statistics
        """
        return {
            "total": self.total_tokens,
            "available": self.available_tokens,
            "used_total": self.get_used(),
            "buffer": self.buffer_tokens,
            "reserved": self.reserved_tokens,
            "used_by_component": self.used_tokens
        }


class TokenControlledRetriever:
    """
    Retriever that controls token usage during document retrieval.
    Ensures retrieved content fits within token budget constraints.
    """
    
    def __init__(self, 
                retriever: Any,
                token_manager: AdaptiveTokenManager,
                token_estimator: TokenEstimator = TokenEstimator()):
        """
        Initialize the retriever.
        
        Args:
            retriever: Underlying retriever
            token_manager: Token manager
            token_estimator: Token estimator
        """
        self.retriever = retriever
        self.token_manager = token_manager
        self.token_estimator = token_estimator
    
    def retrieve(self, query: str, top_k: int = 10, 
                max_tokens: int = None) -> List[Document]:
        """
        Retrieve documents with token control.
        
        Args:
            query: Query string
            top_k: Maximum number of documents to retrieve
            max_tokens: Maximum tokens (None to use token manager)
            
        Returns:
            List of retrieved documents
        """
        # Get token budget from manager if not specified
        if max_tokens is None:
            if "documents" in self.token_manager.components:
                max_tokens = self.token_manager.token_budget.allocations.get("documents", 0)
            else:
                max_tokens = self.token_manager.total_tokens // 2  # Default to half
        
        # Retrieve initial set of documents
        documents = self._retrieve_raw(query, top_k * 2)  # Get more than needed
        
        # Apply token constraints
        selected_docs = []
        total_tokens = 0
        
        for doc in documents:
            doc_tokens = self.token_estimator.estimate_tokens(doc.content)
            
            if total_tokens + doc_tokens <= max_tokens:
                selected_docs.append(doc)
                total_tokens += doc_tokens
            else:
                # Try to include partial document
                remaining = max_tokens - total_tokens
                if remaining > 50:  # Only if meaningful space left
                    truncated_doc = self._truncate_document(doc, remaining)
                    selected_docs.append(truncated_doc)
                    total_tokens += remaining
                break
        
        # Record token usage
        if "documents" in self.token_manager.components:
            self.token_manager.token_budget.use("documents", total_tokens)
        
        return selected_docs
    
    def _retrieve_raw(self, query: str, top_k: int) -> List[Document]:
        """
        Call underlying retriever.
        
        Args:
            query: Query string
            top_k: Maximum number of documents
            
        Returns:
            List of retrieved documents
        """
        # This is a mock implementation that would call the actual retriever
        # In a real system, this would use self.retriever
        
        # Mock documents
        documents = []
        for i in range(top_k):
            doc = Document(
                id=f"doc_{i+1}",
                content=f"This is document {i+1} that might be relevant to the query: {query}. " + 
                       f"It contains various information that could help answer the question.",
                metadata={"source": f"source_{i+1}"},
                score=1.0 - (i * 0.1)
            )
            documents.append(doc)
        
        return documents
    
    def _truncate_document(self, document: Document, max_tokens: int) -> Document:
        """
        Truncate document to fit within token limit.
        
        Args:
            document: Document to truncate
            max_tokens: Maximum tokens
            
        Returns:
            Truncated document
        """
        estimated_chars = max_tokens * 4
        
        # Try sentence boundary
        truncated_content = document.content[:estimated_chars]
        last_period = truncated_content.rfind('. ')
        
        if last_period > estimated_chars * 0.5:
            truncated_content = document.content[:last_period + 1]
        
        # Create new document with truncated content
        truncated_doc = Document(
            id=document.id,
            content=truncated_content,
            metadata={**document.metadata, "truncated": True},
            score=document.score
        )
        
        return truncated_doc


def example_fixed_allocation():
    """Example of fixed allocation strategy."""
    # Define fixed ratios
    ratios = {
        "system": 0.1,      # System prompt
        "history": 0.2,     # Conversation history
        "documents": 0.6,   # Retrieved documents
        "query": 0.05,      # User query
        "examples": 0.05    # Few-shot examples
    }
    
    # Create strategy
    strategy = FixedRatioAllocation(ratios)
    
    # Allocate tokens
    total_tokens = 4000
    components = ["system", "history", "documents", "query", "examples"]
    allocations = strategy.allocate(total_tokens, components)
    
    print("Fixed Allocation Example:")
    print("-----------------------")
    print(f"Total tokens: {total_tokens}")
    for component, allocation in allocations.items():
        print(f"{component}: {allocation} tokens ({allocation/total_tokens:.1%})")


def example_adaptive_manager():
    """Example of adaptive token manager."""
    # Define base allocation
    base_allocation = {
        "system": 0.1,
        "history": 0.2,
        "documents": 0.6,
        "query": 0.05,
        "examples": 0.05
    }
    
    # Create dynamic allocation strategy
    def importance_fn(component, content):
        # Simple mock function
        if component == "query":
            return 1.5  # Higher importance
        elif component == "documents":
            return 1.2  # Higher importance
        else:
            return 1.0  # Standard importance
    
    strategy = DynamicAllocation(base_allocation, importance_fn)
    
    # Create manager
    manager = AdaptiveTokenManager(4000, strategy)
    
    # Set up budget with content
    content = {
        "query": "What is RAG and how can it be optimized?",
        "documents": ["Document about RAG", "Document about optimization"],
        "history": ["User: Hi", "AI: Hello", "User: Tell me about RAG"]
    }
    
    components = ["system", "history", "documents", "query", "examples"]
    budget = manager.setup_budget(components, content)
    
    print("\nAdaptive Token Manager Example:")
    print("-----------------------------")
    print(f"Total budget: {budget.total}")
    for component, allocation in budget.allocations.items():
        print(f"{component}: {allocation} tokens ({allocation/budget.total:.1%})")
    
    # Test fitting content
    long_document = "This is a very long document about Retrieval Augmented Generation (RAG). " * 20
    fitted_document = manager.fit_content("documents", long_document)
    
    print(f"\nOriginal document tokens: {TokenEstimator.estimate_tokens(long_document)}")
    print(f"Fitted document tokens: {TokenEstimator.estimate_tokens(fitted_document)}")
    print(f"Document allocation: {budget.allocations['documents']}")
    
    # Get stats
    stats = manager.get_stats()
    print("\nToken Usage Statistics:")
    print(f"Total used: {stats['total_used']} / {stats['total_budget']}")
    for component, comp_stats in stats["components"].items():
        print(f"{component}: {comp_stats['used']} / {comp_stats['allocation']} tokens used")


def example_token_buffer():
    """Example of token buffer manager."""
    # Create buffer manager
    reserved = {
        "system": 500,
        "examples": 300
    }
    
    buffer_manager = TokenBufferManager(4000, reserved, buffer_tokens=200)
    
    print("\nToken Buffer Manager Example:")
    print("---------------------------")
    print(f"Total tokens: {buffer_manager.total_tokens}")
    print(f"Reserved tokens: {buffer_manager.reserved_tokens}")
    print(f"Buffer tokens: {buffer_manager.buffer_tokens}")
    print(f"Available tokens: {buffer_manager.available_tokens}")
    
    # Allocate tokens
    buffer_manager.allocate("system", 400)
    buffer_manager.allocate("examples", 200)
    buffer_manager.allocate("documents", 2000)
    buffer_manager.allocate("history", 500)
    
    # Get stats
    stats = buffer_manager.get_stats()
    print("\nAfter Allocation:")
    print(f"Available: {stats['available']} tokens")
    print("Used by component:")
    for component, used in stats["used_by_component"].items():
        print(f"  {component}: {used} tokens")
    
    # Release tokens
    buffer_manager.release("documents", 500)
    
    # Get updated stats
    stats = buffer_manager.get_stats()
    print("\nAfter Release:")
    print(f"Available: {stats['available']} tokens")
    print("Used by component:")
    for component, used in stats["used_by_component"].items():
        print(f"  {component}: {used} tokens")


def main():
    """Main function demonstrating token management techniques."""
    print("Token Management for RAG Systems")
    print("===============================")
    
    example_fixed_allocation()
    example_adaptive_manager()
    example_token_buffer()
    
    print("\nToken-Controlled Retriever Example:")
    print("--------------------------------")
    
    # Create token manager
    strategy = FixedRatioAllocation({"documents": 1.0})
    manager = AdaptiveTokenManager(2000, strategy)
    manager.setup_budget(["documents"])
    
    # Create retriever
    retriever = TokenControlledRetriever(None, manager)
    
    # Retrieve with token control
    documents = retriever.retrieve("What is RAG?", max_tokens=1000)
    
    print(f"Retrieved {len(documents)} documents")
    print(f"Total tokens: {sum(TokenEstimator.estimate_tokens(doc.content) for doc in documents)}")
    
    # Show truncation
    if any("truncated" in doc.metadata for doc in documents):
        truncated = next(doc for doc in documents if "truncated" in doc.metadata)
        print(f"\nTruncated document example:")
        print(f"ID: {truncated.id}")
        print(f"Content preview: {truncated.content[:100]}...")


if __name__ == "__main__":
    main() 