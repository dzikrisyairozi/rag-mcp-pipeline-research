"""
Document Q&A System

This module provides a complete implementation of a document Q&A system using
the RAG (Retrieval Augmented Generation) components from the module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import os
import time
import json
import re


@dataclass
class Document:
    """Represents a document in the system."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Any] = None
    chunks: List[Any] = field(default_factory=list)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Any] = None
    doc_id: str = ""
    chunk_index: int = 0


@dataclass
class SearchResult:
    """Represents a search result."""
    chunk: DocumentChunk
    score: float
    rank: int = 0


@dataclass
class QARequest:
    """Represents a question-answering request."""
    query: str
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResponse:
    """Represents a question-answering response."""
    query: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentStore:
    """
    Manages storage and retrieval of documents.
    Includes functionality for document CRUD operations.
    """
    
    def __init__(self, vector_store: Any = None):
        """
        Initialize the document store.
        
        Args:
            vector_store: Vector database for embeddings
        """
        self.documents = {}  # Map from document ID to Document
        self.chunks = {}  # Map from chunk ID to DocumentChunk
        self.vector_store = vector_store
    
    def add_document(self, document: Document) -> bool:
        """
        Add a document to the store.
        
        Args:
            document: Document to add
            
        Returns:
            True if successful, False otherwise
        """
        # Check if document already exists
        if document.id in self.documents:
            return False
        
        # Add document
        self.documents[document.id] = document
        
        # Add chunks if any
        for chunk in document.chunks:
            self.chunks[chunk.id] = chunk
        
        return True
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document or None if not found
        """
        return self.documents.get(doc_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get a chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            DocumentChunk or None if not found
        """
        return self.chunks.get(chunk_id)
    
    def get_documents(self, filter_criteria: Dict[str, Any] = None) -> List[Document]:
        """
        Get documents matching filter criteria.
        
        Args:
            filter_criteria: Criteria to filter documents
            
        Returns:
            List of matching documents
        """
        if not filter_criteria:
            return list(self.documents.values())
        
        # Filter documents based on criteria
        result = []
        
        for doc in self.documents.values():
            matches = True
            
            for key, value in filter_criteria.items():
                if key in doc.metadata:
                    if doc.metadata[key] != value:
                        matches = False
                        break
                else:
                    matches = False
                    break
            
            if matches:
                result.append(doc)
        
        return result
    
    def update_document(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a document.
        
        Args:
            doc_id: Document ID
            updates: Updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        if doc_id not in self.documents:
            return False
        
        doc = self.documents[doc_id]
        
        # Update content if provided
        if "content" in updates:
            doc.content = updates["content"]
        
        # Update metadata if provided
        if "metadata" in updates:
            doc.metadata.update(updates["metadata"])
        
        return True
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        if doc_id not in self.documents:
            return False
        
        # Get document chunks
        doc = self.documents[doc_id]
        
        # Delete chunks
        for chunk in doc.chunks:
            if chunk.id in self.chunks:
                del self.chunks[chunk.id]
        
        # Delete document
        del self.documents[doc_id]
        
        return True
    
    def search_chunks(self, query: str, filters: Dict[str, Any] = None, 
                     top_k: int = 5) -> List[SearchResult]:
        """
        Search for chunks matching a query.
        
        Args:
            query: Search query
            filters: Metadata filters
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if self.vector_store:
            # Use vector store for semantic search
            # This is a mock implementation
            print(f"Searching for '{query}' in vector store")
            
            results = []
            
            # Mock vector search results
            all_chunks = list(self.chunks.values())
            
            # Apply filters if any
            if filters:
                filtered_chunks = []
                
                for chunk in all_chunks:
                    matches = True
                    
                    for key, value in filters.items():
                        if key in chunk.metadata:
                            if chunk.metadata[key] != value:
                                matches = False
                                break
                        else:
                            matches = False
                            break
                    
                    if matches:
                        filtered_chunks.append(chunk)
                
                all_chunks = filtered_chunks
            
            # Sort by mock relevance (random for demonstration)
            import random
            random.shuffle(all_chunks)
            
            # Take top_k results
            for i, chunk in enumerate(all_chunks[:top_k]):
                results.append(SearchResult(
                    chunk=chunk,
                    score=0.9 - (i * 0.1),
                    rank=i + 1
                ))
            
            return results
        else:
            # Fallback to keyword search
            results = []
            
            # Extract query terms
            query_terms = set(query.lower().split())
            
            # Check each chunk
            for chunk in self.chunks.values():
                # Apply filters if any
                if filters:
                    matches = True
                    
                    for key, value in filters.items():
                        if key in chunk.metadata:
                            if chunk.metadata[key] != value:
                                matches = False
                                break
                        else:
                            matches = False
                            break
                    
                    if not matches:
                        continue
                
                # Check content for query terms
                content_lower = chunk.content.lower()
                matches = 0
                
                for term in query_terms:
                    if term in content_lower:
                        matches += 1
                
                if matches > 0:
                    score = matches / len(query_terms)
                    
                    results.append(SearchResult(
                        chunk=chunk,
                        score=score
                    ))
            
            # Sort by score and take top_k
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Assign ranks
            for i, result in enumerate(results[:top_k]):
                result.rank = i + 1
            
            return results[:top_k]


class DocumentProcessor:
    """
    Processes documents for RAG.
    Includes functionality for chunking, embedding, and metadata extraction.
    """
    
    def __init__(self, 
                chunking_strategy: Any,
                embedding_model: Any = None,
                metadata_extractors: List[Any] = None):
        """
        Initialize the document processor.
        
        Args:
            chunking_strategy: Strategy for chunking documents
            embedding_model: Model for creating embeddings
            metadata_extractors: Extractors for document metadata
        """
        self.chunking_strategy = chunking_strategy
        self.embedding_model = embedding_model
        self.metadata_extractors = metadata_extractors or []
    
    def process_document(self, document: Document) -> Document:
        """
        Process a document for RAG.
        
        Args:
            document: Document to process
            
        Returns:
            Processed document
        """
        # Extract metadata
        for extractor in self.metadata_extractors:
            document = extractor.extract_metadata(document)
        
        # Chunk document
        chunks = self.chunk_document(document)
        
        # Create embeddings
        if self.embedding_model:
            self.create_embeddings(chunks)
        
        # Add chunks to document
        document.chunks = chunks
        
        return document
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk a document.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        # Use chunking strategy to create chunks
        # This is a simplified implementation that splits by paragraphs
        chunks = []
        
        # Split content by double newline (paragraphs)
        paragraphs = document.content.split("\n\n")
        
        for i, paragraph in enumerate(paragraphs):
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
            
            # Create chunk
            chunk = DocumentChunk(
                id=f"{document.id}_chunk_{i}",
                content=paragraph,
                metadata=document.metadata.copy(),
                doc_id=document.id,
                chunk_index=i
            )
            
            # Add chunk metadata
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_count"] = len(paragraphs)
            
            chunks.append(chunk)
        
        return chunks
    
    def create_embeddings(self, chunks: List[DocumentChunk]):
        """
        Create embeddings for document chunks.
        
        Args:
            chunks: Document chunks to embed
        """
        if not self.embedding_model:
            return
        
        # Create embeddings
        # This is a mock implementation
        print(f"Creating embeddings for {len(chunks)} chunks")
        
        for chunk in chunks:
            # In a real implementation, this would use the embedding model
            # Mock embedding with random values
            import numpy as np
            chunk.embedding = np.random.rand(768)  # 768-dimensional embedding


class ResponseGenerator:
    """
    Generates answers from retrieved chunks.
    Uses a language model to synthesize coherent responses.
    """
    
    def __init__(self, language_model: Any = None):
        """
        Initialize the response generator.
        
        Args:
            language_model: Language model for generating responses
        """
        self.language_model = language_model
    
    def generate_response(self, query: str, results: List[SearchResult]) -> str:
        """
        Generate a response to a query based on search results.
        
        Args:
            query: User query
            results: Search results
            
        Returns:
            Generated response
        """
        # Prepare context from search results
        context = self._prepare_context(query, results)
        
        # In a real implementation, this would use the language model
        # Mock response generation
        print(f"Generating response for query: {query}")
        
        if not results:
            return "I don't have enough information to answer that question."
        
        # Create a simple response that references the retrieved chunks
        response = f"Based on the information I have, I can answer your question about {query}.\n\n"
        
        # Add information from top results
        for i, result in enumerate(results[:3]):  # Use top 3 results
            chunk = result.chunk
            doc_id = chunk.doc_id
            
            response += f"From source {doc_id}, I found that {chunk.content[:100]}...\n\n"
        
        response += "I hope this answers your question."
        
        return response
    
    def _prepare_context(self, query: str, results: List[SearchResult]) -> str:
        """
        Prepare context for the language model.
        
        Args:
            query: User query
            results: Search results
            
        Returns:
            Formatted context
        """
        context = f"Question: {query}\n\n"
        context += "Context:\n"
        
        for i, result in enumerate(results):
            chunk = result.chunk
            context += f"[{i+1}] {chunk.content}\n\n"
        
        context += "Please answer the question based on the provided context."
        
        return context


class DocumentQASystem:
    """
    Main document Q&A system.
    Integrates document storage, processing, retrieval, and response generation.
    """
    
    def __init__(self, 
                document_store: DocumentStore,
                document_processor: DocumentProcessor,
                context_optimizer: Any,
                response_generator: ResponseGenerator):
        """
        Initialize the document Q&A system.
        
        Args:
            document_store: Store for document storage and retrieval
            document_processor: Processor for document preparation
            context_optimizer: Optimizer for context window management
            response_generator: Generator for answers
        """
        self.document_store = document_store
        self.document_processor = document_processor
        self.context_optimizer = context_optimizer
        self.response_generator = response_generator
        self.conversation_history = []
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a document to the system.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            Document ID
        """
        # Create document
        doc_id = f"doc_{int(time.time())}"
        
        document = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {}
        )
        
        # Process document
        processed_doc = self.document_processor.process_document(document)
        
        # Add to store
        self.document_store.add_document(processed_doc)
        
        return doc_id
    
    def ask(self, request: QARequest) -> QAResponse:
        """
        Answer a question based on the document store.
        
        Args:
            request: Q&A request
            
        Returns:
            Q&A response
        """
        # Search for relevant chunks
        results = self.document_store.search_chunks(
            query=request.query,
            filters=request.filters,
            top_k=request.top_k
        )
        
        # Optimize context
        optimized_results = self._optimize_context(request.query, results)
        
        # Generate response
        answer = self.response_generator.generate_response(request.query, optimized_results)
        
        # Prepare source information
        sources = []
        for result in optimized_results:
            chunk = result.chunk
            doc = self.document_store.get_document(chunk.doc_id)
            
            if doc:
                sources.append({
                    "document_id": doc.id,
                    "chunk_id": chunk.id,
                    "score": result.score,
                    "metadata": chunk.metadata
                })
        
        # Create response
        response = QAResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            metadata={
                "timestamp": time.time(),
                "result_count": len(results)
            }
        )
        
        # Update conversation history
        self._update_history(request, response)
        
        return response
    
    def _optimize_context(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Optimize search results for the context window.
        
        Args:
            query: User query
            results: Search results
            
        Returns:
            Optimized search results
        """
        # In a real implementation, this would use the context optimizer
        # For simplicity, we'll just return the results
        
        # Sort by score
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(sorted_results):
            result.rank = i + 1
        
        return sorted_results
    
    def _update_history(self, request: QARequest, response: QAResponse):
        """
        Update conversation history.
        
        Args:
            request: Q&A request
            response: Q&A response
        """
        self.conversation_history.append({
            "query": request.query,
            "answer": response.answer,
            "timestamp": time.time()
        })
        
        # Keep only the last 10 entries
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_similar_questions(self, query: str, max_questions: int = 3) -> List[Dict[str, Any]]:
        """
        Get similar questions from conversation history.
        
        Args:
            query: Current query
            max_questions: Maximum number of questions to return
            
        Returns:
            List of similar questions with answers
        """
        if not self.conversation_history:
            return []
        
        # Simple similarity metric: word overlap
        query_words = set(query.lower().split())
        
        similar_questions = []
        
        for entry in self.conversation_history:
            history_query = entry["query"]
            history_words = set(history_query.lower().split())
            
            # Calculate similarity
            if not history_words:
                continue
                
            intersection = query_words.intersection(history_words)
            union = query_words.union(history_words)
            
            similarity = len(intersection) / len(union) if union else 0
            
            if similarity > 0.3:  # Threshold for similarity
                similar_questions.append({
                    "query": history_query,
                    "answer": entry["answer"],
                    "similarity": similarity,
                    "timestamp": entry["timestamp"]
                })
        
        # Sort by similarity
        similar_questions.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similar_questions[:max_questions]


def create_sample_qa_system():
    """Create a sample document Q&A system for demonstration."""
    # Create components
    document_store = DocumentStore()
    
    # Mock chunking strategy
    chunking_strategy = "paragraph_chunker"
    
    # Create document processor
    document_processor = DocumentProcessor(chunking_strategy)
    
    # Mock context optimizer
    context_optimizer = "priority_optimizer"
    
    # Create response generator
    response_generator = ResponseGenerator()
    
    # Create Q&A system
    qa_system = DocumentQASystem(
        document_store=document_store,
        document_processor=document_processor,
        context_optimizer=context_optimizer,
        response_generator=response_generator
    )
    
    return qa_system


def demonstrate_document_qa():
    """Demonstrate the document Q&A system."""
    print("Document Q&A System Demonstration")
    print("--------------------------------")
    
    # Create Q&A system
    qa_system = create_sample_qa_system()
    
    # Add sample documents
    print("Adding sample documents...")
    
    doc1_content = """
    Retrieval Augmented Generation (RAG) is an AI framework that enhances large language model (LLM) responses 
    by incorporating knowledge from external sources. In a RAG system, the user's query is used to retrieve relevant 
    information from a knowledge base, and this information is then provided to the LLM as context for generating a response.
    
    RAG addresses several limitations of traditional LLMs:
    1. It provides access to information beyond the model's training data
    2. It reduces hallucinations by grounding responses in retrieved facts
    3. It enables real-time information updates without retraining
    4. It provides citations and sources for generated information
    
    Key components of a RAG system include:
    - Document processing and chunking
    - Embedding models for semantic search
    - Vector databases for efficient retrieval
    - Context window optimization
    - Response generation with citations
    """
    
    doc2_content = """
    Vector databases are specialized database systems designed to store and search vector embeddings efficiently.
    They are a critical component in modern AI systems, particularly for retrieval-based applications.
    
    Popular vector databases include:
    
    1. Pinecone: A fully managed vector database with high performance and scalability.
    2. Weaviate: An open-source vector search engine with classification capabilities.
    3. Chroma: A lightweight embedding database for AI applications.
    4. Qdrant: A vector similarity search engine with extended filtering support.
    5. FAISS (Facebook AI Similarity Search): A library for efficient similarity search.
    
    Vector databases typically use approximate nearest neighbor (ANN) algorithms to enable fast similarity search
    across millions or billions of vectors while maintaining high recall and precision.
    """
    
    doc3_content = """
    Embedding models convert text or other data into numerical vectors that capture semantic meaning.
    These models are trained on large datasets to learn the relationships between words and concepts.
    
    Common embedding models include:
    
    1. OpenAI's text-embedding-ada-002: High-quality embeddings with 1536 dimensions.
    2. Sentence-BERT/MPNet: Open-source models with strong performance on semantic similarity tasks.
    3. BGE Embeddings: Optimized for retrieval in multiple languages.
    4. E5 Models: Specifically designed for asymmetric retrieval (different encoders for queries and documents).
    
    When selecting an embedding model, consider:
    - Dimension size (affects storage requirements and query speed)
    - Semantic quality (ability to capture meaning accurately)
    - Throughput (speed of generating embeddings)
    - Cost (for API-based models)
    """
    
    doc1_id = qa_system.add_document(doc1_content, {"title": "RAG Overview", "topic": "AI"})
    doc2_id = qa_system.add_document(doc2_content, {"title": "Vector Databases", "topic": "Databases"})
    doc3_id = qa_system.add_document(doc3_content, {"title": "Embedding Models", "topic": "AI"})
    
    print(f"Added documents with IDs: {doc1_id}, {doc2_id}, {doc3_id}")
    
    # Ask questions
    questions = [
        "What is RAG and how does it work?",
        "Which vector databases are commonly used?",
        "What should I consider when choosing an embedding model?",
        "How does RAG reduce hallucinations?",
        "Can you compare Pinecone and Weaviate?"
    ]
    
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        print("-" * 50)
        
        # Create request
        request = QARequest(query=question)
        
        # Get answer
        response = qa_system.ask(request)
        
        # Print answer
        print(f"Answer: {response.answer}")
        
        # Print sources
        print("\nSources:")
        for source in response.sources:
            print(f"- {source['document_id']} (Score: {source['score']:.2f})")
        
        # Get similar questions
        if i > 0:
            similar = qa_system.get_similar_questions(question)
            if similar:
                print("\nSimilar questions asked before:")
                for sq in similar:
                    print(f"- {sq['query']}")
    
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    demonstrate_document_qa() 