"""
RAG Integration with Multi-Context Protocol (MCP)

This module provides components for integrating Retrieval Augmented Generation (RAG)
capabilities with the Multi-Context Protocol (MCP) framework, enabling context-aware
retrieval and generation within a structured protocol.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import json
import time


@dataclass
class Document:
    """Represents a document in the retrieval system."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class MCPContext:
    """Represents a context in the Multi-Context Protocol."""
    name: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPMessage:
    """Represents a message in the Multi-Context Protocol."""
    content: str
    contexts: List[MCPContext] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""


class RAGContextType(Enum):
    """Types of RAG-specific contexts in MCP."""
    QUERY = "query"
    RETRIEVED_DOCUMENTS = "retrieved_documents"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    GENERATION_CONFIG = "generation_config"
    RAG_HISTORY = "rag_history"
    TOOL_OUTPUTS = "tool_outputs"
    DOCUMENT_METADATA = "document_metadata"


class MCPRagIntegration:
    """
    Core integration class for RAG and MCP.
    Handles mapping between MCP contexts and RAG operations.
    """
    
    def __init__(self, retrieval_system: Any, generation_system: Any):
        """
        Initialize the MCP-RAG integration.
        
        Args:
            retrieval_system: System for retrieving documents
            generation_system: System for generating responses
        """
        self.retrieval_system = retrieval_system
        self.generation_system = generation_system
        self.conversation_history = []
    
    def process_message(self, message: MCPMessage) -> MCPMessage:
        """
        Process an incoming MCP message.
        
        Args:
            message: Incoming MCP message
            
        Returns:
            Response MCP message
        """
        # Extract query from message content or context
        query = self._extract_query(message)
        
        # Check for existing retrieval context
        retrieved_docs = self._get_retrieved_documents(message)
        
        # If no retrieval context or retrieval is requested, perform retrieval
        if not retrieved_docs or self._should_retrieve(message):
            retrieved_docs = self._perform_retrieval(query, message)
        
        # Generate response
        response_content = self._generate_response(query, retrieved_docs, message)
        
        # Create response contexts
        response_contexts = self._create_response_contexts(query, retrieved_docs, message)
        
        # Update conversation history
        self._update_history(message, response_content, retrieved_docs)
        
        # Create response message
        response = MCPMessage(
            content=response_content,
            contexts=response_contexts,
            metadata={"timestamp": time.time()}
        )
        
        return response
    
    def _extract_query(self, message: MCPMessage) -> str:
        """
        Extract the query from an MCP message.
        
        Args:
            message: MCP message
            
        Returns:
            Query string
        """
        # First check for a query context
        for context in message.contexts:
            if context.name == RAGContextType.QUERY.value:
                return context.data.get("query", "")
        
        # If no query context, use message content
        return message.content
    
    def _get_retrieved_documents(self, message: MCPMessage) -> List[Document]:
        """
        Get retrieved documents from message contexts.
        
        Args:
            message: MCP message
            
        Returns:
            List of retrieved documents or empty list if none
        """
        for context in message.contexts:
            if context.name == RAGContextType.RETRIEVED_DOCUMENTS.value:
                doc_dicts = context.data.get("documents", [])
                return [Document(
                    id=doc.get("id", f"doc_{i}"),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=doc.get("score", 0.0)
                ) for i, doc in enumerate(doc_dicts)]
        
        return []
    
    def _should_retrieve(self, message: MCPMessage) -> bool:
        """
        Determine if retrieval should be performed.
        
        Args:
            message: MCP message
            
        Returns:
            True if retrieval should be performed, False otherwise
        """
        # Check for explicit retrieval requests
        for context in message.contexts:
            if context.name == "command" and context.data.get("action") == "retrieve":
                return True
        
        # Check if message has changed significantly from previous
        if self.conversation_history:
            last_query = self.conversation_history[-1].get("query", "")
            current_query = self._extract_query(message)
            
            # Simple heuristic: if query is similar to previous, don't retrieve again
            # In a real implementation, use semantic similarity
            if self._text_similarity(last_query, current_query) > 0.8:
                return False
        
        return True
    
    def _perform_retrieval(self, query: str, message: MCPMessage) -> List[Document]:
        """
        Perform document retrieval.
        
        Args:
            query: Query string
            message: Original MCP message for context
            
        Returns:
            List of retrieved documents
        """
        # Extract retrieval parameters from contexts
        params = self._extract_retrieval_params(message)
        
        # Use the retrieval system to fetch documents
        # In a real implementation, this would call the actual retrieval system
        print(f"Retrieving documents for query: {query}")
        
        # Mock retrieval for demonstration
        documents = [
            Document(
                id=f"doc_{i+1}",
                content=f"This is document {i+1} about {query}. It contains information relevant to the query.",
                metadata={
                    "source": f"source_{i+1}",
                    "timestamp": time.time()
                },
                score=0.9 - (i * 0.1)
            ) for i in range(params.get("top_k", 3))
        ]
        
        return documents
    
    def _extract_retrieval_params(self, message: MCPMessage) -> Dict[str, Any]:
        """
        Extract retrieval parameters from message contexts.
        
        Args:
            message: MCP message
            
        Returns:
            Dictionary of retrieval parameters
        """
        params = {
            "top_k": 3,
            "min_score": 0.2,
            "use_reranking": True
        }
        
        # Check for retrieval configuration in contexts
        for context in message.contexts:
            if context.name == "retrieval_config":
                # Update params with values from context
                params.update(context.data)
                break
        
        return params
    
    def _generate_response(self, query: str, documents: List[Document], 
                          message: MCPMessage) -> str:
        """
        Generate a response based on the query and retrieved documents.
        
        Args:
            query: Query string
            documents: Retrieved documents
            message: Original MCP message for context
            
        Returns:
            Generated response
        """
        # Extract generation parameters from contexts
        params = self._extract_generation_params(message)
        
        # Prepare context for generation
        generation_context = self._prepare_generation_context(query, documents, message)
        
        # Use the generation system to generate a response
        # In a real implementation, this would call the actual generation system
        print(f"Generating response for query: {query}")
        
        # Mock generation for demonstration
        response = f"Based on the retrieved information, I can tell you that {query} is an important concept in the field. "
        response += "The documents indicate several key aspects: "
        
        for i, doc in enumerate(documents[:2]):  # Use only first 2 docs for brevity
            response += f"Document {i+1} mentions that it contains information about {query}. "
        
        if params.get("verbose", False):
            response += "\n\nThis information was compiled from multiple sources in our knowledge base."
        
        return response
    
    def _extract_generation_params(self, message: MCPMessage) -> Dict[str, Any]:
        """
        Extract generation parameters from message contexts.
        
        Args:
            message: MCP message
            
        Returns:
            Dictionary of generation parameters
        """
        params = {
            "temperature": 0.7,
            "max_tokens": 500,
            "verbose": False
        }
        
        # Check for generation configuration in contexts
        for context in message.contexts:
            if context.name == RAGContextType.GENERATION_CONFIG.value:
                # Update params with values from context
                params.update(context.data)
                break
        
        return params
    
    def _prepare_generation_context(self, query: str, documents: List[Document], 
                                   message: MCPMessage) -> str:
        """
        Prepare the context for generation.
        
        Args:
            query: Query string
            documents: Retrieved documents
            message: Original MCP message for context
            
        Returns:
            Formatted context string
        """
        # Start with query
        context = f"Query: {query}\n\n"
        
        # Add document contents
        context += "Retrieved Documents:\n"
        for i, doc in enumerate(documents):
            context += f"[{i+1}] {doc.content}\n\n"
        
        # Add history if available
        history_context = self._format_history()
        if history_context:
            context += f"Conversation History:\n{history_context}\n\n"
        
        return context
    
    def _create_response_contexts(self, query: str, documents: List[Document], 
                                 message: MCPMessage) -> List[MCPContext]:
        """
        Create contexts for the response message.
        
        Args:
            query: Query string
            documents: Retrieved documents
            message: Original MCP message
            
        Returns:
            List of response contexts
        """
        contexts = []
        
        # Add retrieved documents context
        doc_dicts = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": doc.score
            }
            for doc in documents
        ]
        
        contexts.append(MCPContext(
            name=RAGContextType.RETRIEVED_DOCUMENTS.value,
            data={"documents": doc_dicts},
            metadata={"timestamp": time.time()}
        ))
        
        # Add history context
        history_entries = [
            {
                "query": entry["query"],
                "response": entry["response"],
                "timestamp": entry["timestamp"]
            }
            for entry in self.conversation_history[-5:]  # Last 5 entries
        ]
        
        contexts.append(MCPContext(
            name=RAGContextType.RAG_HISTORY.value,
            data={"history": history_entries},
            metadata={"timestamp": time.time()}
        ))
        
        # Add document metadata context
        doc_metadata = [
            {
                "id": doc.id,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
        
        contexts.append(MCPContext(
            name=RAGContextType.DOCUMENT_METADATA.value,
            data={"document_metadata": doc_metadata},
            metadata={"timestamp": time.time()}
        ))
        
        return contexts
    
    def _update_history(self, message: MCPMessage, response: str, 
                       documents: List[Document]):
        """
        Update conversation history.
        
        Args:
            message: Original MCP message
            response: Generated response
            documents: Retrieved documents
        """
        query = self._extract_query(message)
        
        self.conversation_history.append({
            "query": query,
            "response": response,
            "documents": [doc.id for doc in documents],
            "timestamp": time.time()
        })
        
        # Keep only the last 10 entries
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def _format_history(self) -> str:
        """
        Format conversation history for context.
        
        Returns:
            Formatted history string
        """
        if not self.conversation_history:
            return ""
        
        history_str = ""
        for i, entry in enumerate(self.conversation_history[-3:]):  # Last 3 entries
            history_str += f"User: {entry['query']}\n"
            history_str += f"Assistant: {entry['response']}\n\n"
        
        return history_str
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity (jaccard similarity of words).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union


class RAGContextManager:
    """
    Manages RAG-specific contexts in the MCP framework.
    Handles context hierarchy and stateful conversations.
    """
    
    def __init__(self):
        """Initialize the RAG context manager."""
        self.contexts = {}  # Maps context names to their current values
        self.context_hierarchy = {
            RAGContextType.QUERY.value: 0,
            RAGContextType.RETRIEVED_DOCUMENTS.value: 1,
            RAGContextType.KNOWLEDGE_GRAPH.value: 1,
            RAGContextType.GENERATION_CONFIG.value: 2,
            RAGContextType.RAG_HISTORY.value: 3,
            RAGContextType.TOOL_OUTPUTS.value: 2,
            RAGContextType.DOCUMENT_METADATA.value: 2
        }
    
    def update_context(self, context: MCPContext):
        """
        Update a context with new data.
        
        Args:
            context: New context to add or update
        """
        self.contexts[context.name] = context
    
    def get_context(self, context_name: str) -> Optional[MCPContext]:
        """
        Get a context by name.
        
        Args:
            context_name: Name of the context to get
            
        Returns:
            The context or None if not found
        """
        return self.contexts.get(context_name)
    
    def get_all_contexts(self) -> List[MCPContext]:
        """
        Get all current contexts.
        
        Returns:
            List of all contexts
        """
        return list(self.contexts.values())
    
    def get_contexts_by_level(self, level: int) -> List[MCPContext]:
        """
        Get contexts at a specific hierarchy level.
        
        Args:
            level: Hierarchy level
            
        Returns:
            List of contexts at that level
        """
        return [
            context for name, context in self.contexts.items()
            if self.context_hierarchy.get(name, 0) == level
        ]
    
    def clear_contexts(self):
        """Clear all contexts."""
        self.contexts = {}
    
    def create_query_context(self, query: str) -> MCPContext:
        """
        Create a query context.
        
        Args:
            query: Query string
            
        Returns:
            Query context
        """
        context = MCPContext(
            name=RAGContextType.QUERY.value,
            data={"query": query},
            metadata={"timestamp": time.time()}
        )
        
        self.update_context(context)
        return context
    
    def create_documents_context(self, documents: List[Document]) -> MCPContext:
        """
        Create a retrieved documents context.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Retrieved documents context
        """
        doc_dicts = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": doc.score
            }
            for doc in documents
        ]
        
        context = MCPContext(
            name=RAGContextType.RETRIEVED_DOCUMENTS.value,
            data={"documents": doc_dicts},
            metadata={"timestamp": time.time()}
        )
        
        self.update_context(context)
        return context


class StatefulRAGSession:
    """
    Manages a stateful RAG session within the MCP framework.
    Maintains conversation history and handles context persistence.
    """
    
    def __init__(self, integration: MCPRagIntegration, session_id: str = ""):
        """
        Initialize the stateful RAG session.
        
        Args:
            integration: MCP-RAG integration instance
            session_id: Unique session identifier
        """
        self.integration = integration
        self.session_id = session_id or f"session_{time.time()}"
        self.context_manager = RAGContextManager()
        self.message_history = []
    
    def process_message(self, message: MCPMessage) -> MCPMessage:
        """
        Process a message in the current session.
        
        Args:
            message: Incoming message
            
        Returns:
            Response message
        """
        # Add message to history
        self.message_history.append({"role": "user", "message": message})
        
        # Update contexts from message
        self._update_contexts_from_message(message)
        
        # Process message with integration
        response = self.integration.process_message(message)
        
        # Update contexts from response
        self._update_contexts_from_message(response)
        
        # Add response to history
        self.message_history.append({"role": "assistant", "message": response})
        
        return response
    
    def _update_contexts_from_message(self, message: MCPMessage):
        """
        Update contexts based on a message.
        
        Args:
            message: MCP message
        """
        for context in message.contexts:
            self.context_manager.update_context(context)
    
    def get_session_state(self) -> Dict[str, Any]:
        """
        Get the current session state.
        
        Returns:
            Dictionary with session state
        """
        return {
            "session_id": self.session_id,
            "contexts": {name: context.data for name, context in self.context_manager.contexts.items()},
            "message_count": len(self.message_history)
        }
    
    def create_message(self, content: str, include_contexts: bool = True) -> MCPMessage:
        """
        Create a new message with current contexts.
        
        Args:
            content: Message content
            include_contexts: Whether to include current contexts
            
        Returns:
            New MCP message
        """
        contexts = []
        
        if include_contexts:
            contexts = self.context_manager.get_all_contexts()
        
        return MCPMessage(
            content=content,
            contexts=contexts,
            metadata={"session_id": self.session_id, "timestamp": time.time()}
        )


def create_sample_documents(n: int = 3) -> List[Document]:
    """Create sample documents for testing."""
    documents = []
    
    topics = [
        "Retrieval Augmented Generation",
        "Vector Databases",
        "Embedding Models",
        "Context Window Optimization",
        "Multi-Context Protocol"
    ]
    
    for i in range(n):
        topic = topics[i % len(topics)]
        
        doc = Document(
            id=f"doc_{i+1}",
            content=f"This document covers {topic}. It explains key concepts and applications in this field.",
            metadata={
                "topic": topic,
                "source": f"source_{i+1}",
                "timestamp": time.time()
            },
            score=0.9 - (i * 0.1)
        )
        
        documents.append(doc)
    
    return documents


def demonstrate_mcp_rag_integration():
    """Demonstrate the MCP-RAG integration."""
    print("RAG Integration with Multi-Context Protocol (MCP)")
    print("------------------------------------------------")
    
    # Create mock retrieval and generation systems
    retrieval_system = "Mock Retrieval System"
    generation_system = "Mock Generation System"
    
    # Create integration
    integration = MCPRagIntegration(retrieval_system, generation_system)
    
    # Create stateful session
    session = StatefulRAGSession(integration)
    
    # Process a series of messages
    queries = [
        "What is retrieval augmented generation?",
        "How does it compare to regular language models?",
        "What are the benefits of using RAG?"
    ]
    
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: {query}")
        print("-" * 50)
        
        # Create message
        message = session.create_message(query)
        
        # Add a specific context for demonstration
        if i == 1:
            # Add a generation config context
            gen_config = MCPContext(
                name=RAGContextType.GENERATION_CONFIG.value,
                data={"temperature": 0.5, "verbose": True},
                metadata={"timestamp": time.time()}
            )
            message.contexts.append(gen_config)
        
        # Process message
        response = session.process_message(message)
        
        # Print response
        print(f"Response: {response.content}")
        
        # Print contexts
        print("\nResponse Contexts:")
        for context in response.contexts:
            print(f"- {context.name}: {len(context.data.keys())} keys")
    
    # Print session state
    print("\nFinal Session State:")
    state = session.get_session_state()
    print(f"Session ID: {state['session_id']}")
    print(f"Message Count: {state['message_count']}")
    print(f"Active Contexts: {', '.join(state['contexts'].keys())}")


if __name__ == "__main__":
    demonstrate_mcp_rag_integration() 