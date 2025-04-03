"""
Augmentation Strategies for RAG Systems

This module provides implementations of various augmentation strategies beyond simple retrieval,
including tool augmentation, knowledge graph integration, and self-query refinement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import json
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
class ToolResult:
    """Represents the result of a tool invocation."""
    tool_name: str
    output: Any
    error: Optional[str] = None
    success: bool = True
    execution_time: float = 0.0


@dataclass
class KnowledgeGraphNode:
    """Represents a node in a knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    node_type: str = "entity"


@dataclass
class KnowledgeGraphEdge:
    """Represents an edge (relationship) in a knowledge graph."""
    source_id: str
    target_id: str
    relationship: str
    properties: Dict[str, Any] = field(default_factory=dict)


class AugmentationStrategy(ABC):
    """Abstract base class for augmentation strategies."""
    
    @abstractmethod
    def augment(self, query: str, retrieved_documents: List[Document] = None) -> Dict[str, Any]:
        """
        Augment the context for a query.
        
        Args:
            query: The user query
            retrieved_documents: List of already retrieved documents (optional)
            
        Returns:
            Dictionary containing augmented context
        """
        pass


class ToolAugmentation(AugmentationStrategy):
    """
    Augments context by invoking relevant tools based on the query.
    Useful for accessing external systems, APIs, or performing computations.
    """
    
    def __init__(self, tools: Dict[str, Callable]):
        """
        Initialize the tool augmentation strategy.
        
        Args:
            tools: Dictionary mapping tool names to callable functions
        """
        self.tools = tools
    
    def augment(self, query: str, retrieved_documents: List[Document] = None) -> Dict[str, Any]:
        """Augment context by invoking relevant tools."""
        print(f"Augmenting with tools for query: {query}")
        
        # Identify relevant tools for the query
        relevant_tools = self._identify_relevant_tools(query)
        
        # Invoke relevant tools
        tool_results = []
        
        for tool_name in relevant_tools:
            if tool_name in self.tools:
                result = self._invoke_tool(tool_name, query, retrieved_documents)
                tool_results.append(result)
        
        # Format results as augmented context
        augmented_context = {
            "source": "tool_augmentation",
            "query": query,
            "tool_results": tool_results
        }
        
        return augmented_context
    
    def _identify_relevant_tools(self, query: str) -> List[str]:
        """
        Identify which tools are relevant for the query.
        
        In a real implementation, this would use a classifier or heuristic rules.
        Here we use a simple keyword matching approach for demonstration.
        """
        relevant_tools = []
        
        # Simple keyword matching
        if "calculate" in query.lower() or "compute" in query.lower():
            relevant_tools.append("calculator")
        
        if "weather" in query.lower() or "temperature" in query.lower():
            relevant_tools.append("weather")
        
        if "news" in query.lower() or "latest" in query.lower():
            relevant_tools.append("news")
        
        if "search" in query.lower() or "find" in query.lower():
            relevant_tools.append("search")
        
        # If no specific tool identified, use a default
        if not relevant_tools and "tool_default" in self.tools:
            relevant_tools.append("tool_default")
        
        return relevant_tools
    
    def _invoke_tool(self, tool_name: str, query: str, 
                    retrieved_documents: List[Document] = None) -> ToolResult:
        """
        Invoke a tool with the query.
        
        Args:
            tool_name: Name of the tool to invoke
            query: The user query
            retrieved_documents: List of already retrieved documents (optional)
            
        Returns:
            Result of the tool invocation
        """
        print(f"Invoking tool: {tool_name}")
        
        try:
            # Record start time
            start_time = time.time()
            
            # Call the tool function
            tool_func = self.tools[tool_name]
            
            # Pass retrieved documents if the tool can use them
            if retrieved_documents:
                output = tool_func(query, retrieved_documents)
            else:
                output = tool_func(query)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            return ToolResult(
                tool_name=tool_name,
                output=output,
                success=True,
                execution_time=execution_time
            )
        
        except Exception as e:
            # Log the error and return a failed result
            error_message = str(e)
            print(f"Tool {tool_name} failed: {error_message}")
            
            return ToolResult(
                tool_name=tool_name,
                output=None,
                error=error_message,
                success=False,
                execution_time=0.0
            )


class KnowledgeGraphAugmentation(AugmentationStrategy):
    """
    Augments context by retrieving relevant information from a knowledge graph.
    Useful for structured data and relationships between entities.
    """
    
    def __init__(self, 
                nodes: Dict[str, KnowledgeGraphNode], 
                edges: List[KnowledgeGraphEdge],
                entity_extractor: Optional[Callable] = None):
        """
        Initialize the knowledge graph augmentation strategy.
        
        Args:
            nodes: Dictionary mapping node IDs to nodes
            edges: List of edges (relationships)
            entity_extractor: Optional function to extract entities from text
        """
        self.nodes = nodes
        self.edges = edges
        self.entity_extractor = entity_extractor
        
        # Create adjacency lists for efficient traversal
        self._create_adjacency_lists()
    
    def _create_adjacency_lists(self):
        """Create adjacency lists for the graph for efficient traversal."""
        # Outgoing edges for each node
        self.outgoing = {}
        # Incoming edges for each node
        self.incoming = {}
        
        for edge in self.edges:
            # Add to outgoing edges
            if edge.source_id not in self.outgoing:
                self.outgoing[edge.source_id] = []
            self.outgoing[edge.source_id].append((edge.target_id, edge.relationship, edge.properties))
            
            # Add to incoming edges
            if edge.target_id not in self.incoming:
                self.incoming[edge.target_id] = []
            self.incoming[edge.target_id].append((edge.source_id, edge.relationship, edge.properties))
    
    def augment(self, query: str, retrieved_documents: List[Document] = None) -> Dict[str, Any]:
        """Augment context by retrieving knowledge graph information."""
        print(f"Augmenting with knowledge graph for query: {query}")
        
        # Extract entities from query and documents
        entities = self._extract_entities(query, retrieved_documents)
        
        # Retrieve relevant subgraph
        subgraph = self._retrieve_subgraph(entities)
        
        # Format results as augmented context
        augmented_context = {
            "source": "knowledge_graph_augmentation",
            "query": query,
            "entities": entities,
            "subgraph": {
                "nodes": [self.nodes[node_id] for node_id in subgraph["node_ids"] if node_id in self.nodes],
                "edges": subgraph["edges"]
            }
        }
        
        return augmented_context
    
    def _extract_entities(self, query: str, 
                         retrieved_documents: List[Document] = None) -> List[str]:
        """
        Extract entities from query and documents.
        
        Args:
            query: The user query
            retrieved_documents: List of already retrieved documents (optional)
            
        Returns:
            List of entity IDs
        """
        # If an entity extractor is provided, use it
        if self.entity_extractor:
            # Combine query and document text
            text = query
            if retrieved_documents:
                for doc in retrieved_documents:
                    text += " " + doc.content
            
            # Extract entities
            return self.entity_extractor(text)
        
        # Fallback: Simple string matching against node labels
        entities = []
        
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        for node_id, node in self.nodes.items():
            if node.label.lower() in query_lower:
                entities.append(node_id)
        
        # Also check document content if available
        if retrieved_documents:
            for doc in retrieved_documents:
                doc_lower = doc.content.lower()
                for node_id, node in self.nodes.items():
                    if node.label.lower() in doc_lower and node_id not in entities:
                        entities.append(node_id)
        
        return entities
    
    def _retrieve_subgraph(self, entity_ids: List[str], 
                          max_depth: int = 2) -> Dict[str, Any]:
        """
        Retrieve a subgraph around the given entities.
        
        Args:
            entity_ids: List of entity IDs to start from
            max_depth: Maximum traversal depth
            
        Returns:
            Dictionary containing node IDs and edges in the subgraph
        """
        # Set of node IDs in the subgraph
        node_ids = set(entity_ids)
        # Set of edges in the subgraph
        edges = []
        
        # Queue for BFS traversal
        # Each entry is (node_id, current_depth)
        queue = [(node_id, 0) for node_id in entity_ids]
        # Set of visited nodes
        visited = set(entity_ids)
        
        while queue:
            node_id, depth = queue.pop(0)
            
            # If reached max depth, don't traverse further
            if depth >= max_depth:
                continue
            
            # Process outgoing edges
            if node_id in self.outgoing:
                for target_id, relationship, properties in self.outgoing[node_id]:
                    # Add edge to subgraph
                    edge = KnowledgeGraphEdge(
                        source_id=node_id,
                        target_id=target_id,
                        relationship=relationship,
                        properties=properties
                    )
                    edges.append(edge)
                    
                    # Add target node to subgraph
                    node_ids.add(target_id)
                    
                    # Enqueue target node if not visited
                    if target_id not in visited:
                        visited.add(target_id)
                        queue.append((target_id, depth + 1))
            
            # Process incoming edges
            if node_id in self.incoming:
                for source_id, relationship, properties in self.incoming[node_id]:
                    # Add edge to subgraph
                    edge = KnowledgeGraphEdge(
                        source_id=source_id,
                        target_id=node_id,
                        relationship=relationship,
                        properties=properties
                    )
                    edges.append(edge)
                    
                    # Add source node to subgraph
                    node_ids.add(source_id)
                    
                    # Enqueue source node if not visited
                    if source_id not in visited:
                        visited.add(source_id)
                        queue.append((source_id, depth + 1))
        
        return {
            "node_ids": list(node_ids),
            "edges": edges
        }


class SelfQueryRefinement(AugmentationStrategy):
    """
    Uses the LLM to refine its own query for better retrieval.
    Improves retrieval by generating more effective search queries.
    """
    
    def __init__(self, 
                query_refiner: Optional[Callable] = None,
                retrieval_function: Optional[Callable] = None):
        """
        Initialize the self-query refinement strategy.
        
        Args:
            query_refiner: Function to refine queries (if None, uses a mock implementation)
            retrieval_function: Function to retrieve documents using refined queries
        """
        self.query_refiner = query_refiner
        self.retrieval_function = retrieval_function
    
    def augment(self, query: str, retrieved_documents: List[Document] = None) -> Dict[str, Any]:
        """Augment context by refining the query and retrieving better results."""
        print(f"Augmenting with self-query refinement for query: {query}")
        
        # Generate refined queries
        refined_queries = self._generate_refined_queries(query, retrieved_documents)
        
        # Retrieve documents for each refined query
        all_results = []
        
        for refined_query in refined_queries:
            # Use the retrieval function if provided
            if self.retrieval_function:
                results = self.retrieval_function(refined_query)
            else:
                # Mock implementation for demonstration
                results = self._mock_retrieve(refined_query)
            
            all_results.append({
                "refined_query": refined_query,
                "results": results
            })
        
        # Format results as augmented context
        augmented_context = {
            "source": "self_query_refinement",
            "original_query": query,
            "refined_queries": refined_queries,
            "retrieval_results": all_results
        }
        
        return augmented_context
    
    def _generate_refined_queries(self, query: str, 
                                retrieved_documents: List[Document] = None) -> List[str]:
        """
        Generate refined queries based on the original query.
        
        Args:
            query: Original query
            retrieved_documents: List of already retrieved documents (optional)
            
        Returns:
            List of refined queries
        """
        # Use the query refiner if provided
        if self.query_refiner:
            return self.query_refiner(query, retrieved_documents)
        
        # Mock implementation for demonstration
        print("Using mock query refinement")
        
        # Extract key terms
        terms = re.findall(r'\b\w+\b', query.lower())
        terms = [term for term in terms if len(term) > 3]  # Filter out short terms
        
        # Generate variations
        refined_queries = [query]  # Include original query
        
        # Add synonyms for key terms
        if "algorithm" in terms:
            refined_queries.append(query.replace("algorithm", "method"))
        if "example" in terms:
            refined_queries.append(query.replace("example", "sample code"))
        if "difference" in terms or "between" in terms:
            refined_queries.append(f"compare {query}")
        
        # Add specificity
        refined_queries.append(f"{query} step by step")
        refined_queries.append(f"{query} detailed explanation")
        
        # Remove duplicates
        refined_queries = list(set(refined_queries))
        
        return refined_queries
    
    def _mock_retrieve(self, query: str) -> List[Document]:
        """Mock document retrieval for demonstration."""
        # Generate a few mock documents
        mock_docs = []
        
        for i in range(3):
            mock_docs.append(Document(
                id=f"doc_{i+1}_{hash(query) % 100}",
                content=f"This is a mock document {i+1} for query: {query}.",
                score=0.9 - (i * 0.1)
            ))
        
        return mock_docs


# Example implementations for demonstration

def calculator_tool(query: str) -> Any:
    """Example calculator tool that extracts and evaluates simple expressions."""
    print(f"Running calculator tool on: {query}")
    
    # Extract numeric expressions (very simple implementation)
    expressions = re.findall(r'\d+\s*[\+\-\*\/]\s*\d+', query)
    
    if not expressions:
        return {"result": "No mathematical expression found"}
    
    # Evaluate expressions
    results = {}
    for expr in expressions:
        try:
            # This is unsafe for real applications; use a safer evaluation method
            result = eval(expr)
            results[expr] = result
        except Exception as e:
            results[expr] = f"Error: {str(e)}"
    
    return {"expressions": results}


def weather_tool(query: str) -> Any:
    """Example weather tool that returns mock weather data."""
    print(f"Running weather tool on: {query}")
    
    # Extract location (simple implementation)
    locations = re.findall(r'in\s+([A-Za-z\s]+)', query)
    
    if not locations:
        return {"result": "No location found in query"}
    
    # Generate mock weather data
    weather_data = {}
    for location in locations:
        location = location.strip()
        weather_data[location] = {
            "temperature": f"{65 + hash(location) % 30}°F",
            "condition": ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"][hash(location) % 4],
            "humidity": f"{40 + hash(location) % 40}%",
            "wind": f"{5 + hash(location) % 15} mph"
        }
    
    return {"weather": weather_data}


def create_sample_knowledge_graph() -> Tuple[Dict[str, KnowledgeGraphNode], List[KnowledgeGraphEdge]]:
    """Create a sample knowledge graph for demonstration."""
    # Create nodes
    nodes = {
        "ml": KnowledgeGraphNode(
            id="ml",
            label="Machine Learning",
            properties={"definition": "Field of AI that enables systems to learn from data"},
            node_type="concept"
        ),
        "nlp": KnowledgeGraphNode(
            id="nlp",
            label="Natural Language Processing",
            properties={"definition": "Branch of AI dealing with human language"},
            node_type="concept"
        ),
        "dl": KnowledgeGraphNode(
            id="dl",
            label="Deep Learning",
            properties={"definition": "Subset of ML using neural networks with multiple layers"},
            node_type="concept"
        ),
        "rag": KnowledgeGraphNode(
            id="rag",
            label="Retrieval Augmented Generation",
            properties={"definition": "Technique that enhances LLMs with external knowledge"},
            node_type="concept"
        ),
        "llm": KnowledgeGraphNode(
            id="llm",
            label="Large Language Model",
            properties={"definition": "AI model trained on vast text data to generate human-like text"},
            node_type="technology"
        ),
        "bert": KnowledgeGraphNode(
            id="bert",
            label="BERT",
            properties={"full_name": "Bidirectional Encoder Representations from Transformers"},
            node_type="model"
        ),
        "gpt": KnowledgeGraphNode(
            id="gpt",
            label="GPT",
            properties={"full_name": "Generative Pre-trained Transformer"},
            node_type="model"
        )
    }
    
    # Create edges
    edges = [
        KnowledgeGraphEdge(
            source_id="ml",
            target_id="dl",
            relationship="has_subset",
            properties={"strength": "strong"}
        ),
        KnowledgeGraphEdge(
            source_id="ml",
            target_id="nlp",
            relationship="has_application",
            properties={"strength": "medium"}
        ),
        KnowledgeGraphEdge(
            source_id="dl",
            target_id="llm",
            relationship="enables",
            properties={"strength": "strong"}
        ),
        KnowledgeGraphEdge(
            source_id="nlp",
            target_id="bert",
            relationship="uses",
            properties={"strength": "strong"}
        ),
        KnowledgeGraphEdge(
            source_id="llm",
            target_id="gpt",
            relationship="has_example",
            properties={"strength": "strong"}
        ),
        KnowledgeGraphEdge(
            source_id="rag",
            target_id="llm",
            relationship="augments",
            properties={"strength": "strong"}
        )
    ]
    
    return nodes, edges


def demonstrate_augmentation_strategies():
    """Demonstrate different augmentation strategies."""
    print("Augmentation Strategies for RAG Systems")
    print("--------------------------------------")
    
    # Create tool augmentation strategy
    tools = {
        "calculator": calculator_tool,
        "weather": weather_tool
    }
    tool_augmentation = ToolAugmentation(tools)
    
    # Create knowledge graph augmentation strategy
    nodes, edges = create_sample_knowledge_graph()
    kg_augmentation = KnowledgeGraphAugmentation(nodes, edges)
    
    # Create self-query refinement strategy
    query_refinement = SelfQueryRefinement()
    
    # Example queries
    queries = [
        "What is 24 + 36?",
        "What's the weather in New York?",
        "Explain how BERT and GPT are related to machine learning",
        "What are the best practices for implementing RAG systems?",
    ]
    
    # Test each query with each strategy
    for query in queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("-" * 80)
        
        # Apply tool augmentation
        print("\nTool Augmentation:")
        tool_result = tool_augmentation.augment(query)
        print(f"Result: {json.dumps(tool_result, indent=2)[:300]}...")
        
        # Apply knowledge graph augmentation
        print("\nKnowledge Graph Augmentation:")
        kg_result = kg_augmentation.augment(query)
        print(f"Found {len(kg_result['entities'])} entities and {len(kg_result['subgraph']['edges'])} relationships")
        
        # Apply self-query refinement
        print("\nSelf-Query Refinement:")
        refinement_result = query_refinement.augment(query)
        print(f"Generated {len(refinement_result['refined_queries'])} refined queries")
        
        print("=" * 80)


if __name__ == "__main__":
    demonstrate_augmentation_strategies() 