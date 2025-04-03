"""
Knowledge Graph Integration for RAG Systems

This module provides implementations for integrating knowledge graphs with RAG systems,
enhancing retrieval and reasoning capabilities with structured knowledge.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import json
import re
import time


@dataclass
class KGNode:
    """Represents a node in a knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    node_type: str = "entity"
    
    def __eq__(self, other):
        if not isinstance(other, KGNode):
            return False
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class KGEdge:
    """Represents an edge (relationship) in a knowledge graph."""
    source_id: str
    target_id: str
    relationship: str
    properties: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"{self.source_id}_{self.relationship}_{self.target_id}"
    
    def __eq__(self, other):
        if not isinstance(other, KGEdge):
            return False
        return (
            self.source_id == other.source_id and
            self.target_id == other.target_id and
            self.relationship == other.relationship
        )
    
    def __hash__(self):
        return hash((self.source_id, self.relationship, self.target_id))


@dataclass
class KGTriple:
    """Represents a triple (subject, predicate, object) in a knowledge graph."""
    subject: KGNode
    predicate: str
    object: KGNode
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_edge(self) -> KGEdge:
        """Convert triple to edge."""
        return KGEdge(
            source_id=self.subject.id,
            target_id=self.object.id,
            relationship=self.predicate,
            properties=self.properties
        )


class KnowledgeGraph:
    """
    Represents a knowledge graph with nodes and edges.
    Provides methods for traversal, querying, and integration with RAG systems.
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize an empty knowledge graph.
        
        Args:
            name: Name of the knowledge graph
        """
        self.name = name
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []
        
        # Index for efficient traversal
        self.outgoing: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
        self.incoming: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
        
        # Index node labels for efficient lookup
        self.label_to_nodes: Dict[str, List[str]] = {}
    
    def add_node(self, node: KGNode) -> bool:
        """
        Add a node to the knowledge graph.
        
        Args:
            node: Node to add
            
        Returns:
            True if added, False if node already exists
        """
        # Check if node already exists
        if node.id in self.nodes:
            return False
        
        # Add node
        self.nodes[node.id] = node
        
        # Update label index
        if node.label not in self.label_to_nodes:
            self.label_to_nodes[node.label] = []
        self.label_to_nodes[node.label].append(node.id)
        
        # Initialize traversal indices
        if node.id not in self.outgoing:
            self.outgoing[node.id] = []
        if node.id not in self.incoming:
            self.incoming[node.id] = []
        
        return True
    
    def add_edge(self, edge: KGEdge) -> bool:
        """
        Add an edge to the knowledge graph.
        
        Args:
            edge: Edge to add
            
        Returns:
            True if added, False if source or target node doesn't exist or edge already exists
        """
        # Check if source and target nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            return False
        
        # Check if edge already exists
        if edge in self.edges:
            return False
        
        # Add edge
        self.edges.append(edge)
        
        # Update traversal indices
        self.outgoing[edge.source_id].append((edge.target_id, edge.relationship, edge.properties))
        self.incoming[edge.target_id].append((edge.source_id, edge.relationship, edge.properties))
        
        return True
    
    def add_triple(self, triple: KGTriple) -> bool:
        """
        Add a triple to the knowledge graph.
        
        Args:
            triple: Triple to add
            
        Returns:
            True if added, False otherwise
        """
        # Add subject and object nodes
        self.add_node(triple.subject)
        self.add_node(triple.object)
        
        # Add edge
        edge = triple.to_edge()
        return self.add_edge(edge)
    
    def get_node(self, node_id: str) -> Optional[KGNode]:
        """
        Get a node by ID.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Node or None if not found
        """
        return self.nodes.get(node_id)
    
    def get_nodes_by_label(self, label: str) -> List[KGNode]:
        """
        Get nodes by label.
        
        Args:
            label: Label to search for
            
        Returns:
            List of nodes with the given label
        """
        node_ids = self.label_to_nodes.get(label, [])
        return [self.nodes[node_id] for node_id in node_ids]
    
    def get_nodes_by_type(self, node_type: str) -> List[KGNode]:
        """
        Get nodes by type.
        
        Args:
            node_type: Type to search for
            
        Returns:
            List of nodes with the given type
        """
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def get_edges(self, source_id: Optional[str] = None, 
                 target_id: Optional[str] = None, 
                 relationship: Optional[str] = None) -> List[KGEdge]:
        """
        Get edges by source, target, and/or relationship.
        
        Args:
            source_id: Optional source node ID
            target_id: Optional target node ID
            relationship: Optional relationship type
            
        Returns:
            List of matching edges
        """
        result = []
        
        for edge in self.edges:
            matches = True
            
            if source_id is not None and edge.source_id != source_id:
                matches = False
            if target_id is not None and edge.target_id != target_id:
                matches = False
            if relationship is not None and edge.relationship != relationship:
                matches = False
            
            if matches:
                result.append(edge)
        
        return result
    
    def get_neighbors(self, node_id: str, direction: str = "outgoing") -> List[KGNode]:
        """
        Get neighboring nodes.
        
        Args:
            node_id: ID of the node
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            List of neighboring nodes
        """
        neighbors = []
        
        if direction in ["outgoing", "both"] and node_id in self.outgoing:
            for target_id, _, _ in self.outgoing[node_id]:
                neighbor = self.nodes.get(target_id)
                if neighbor and neighbor not in neighbors:
                    neighbors.append(neighbor)
        
        if direction in ["incoming", "both"] and node_id in self.incoming:
            for source_id, _, _ in self.incoming[node_id]:
                neighbor = self.nodes.get(source_id)
                if neighbor and neighbor not in neighbors:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def get_subgraph(self, seed_node_ids: List[str], max_depth: int = 2) -> 'KnowledgeGraph':
        """
        Extract a subgraph around seed nodes.
        
        Args:
            seed_node_ids: IDs of seed nodes
            max_depth: Maximum traversal depth
            
        Returns:
            Subgraph as a new KnowledgeGraph
        """
        # Create new graph
        subgraph = KnowledgeGraph(f"{self.name}_subgraph")
        
        # Set of node IDs in the subgraph
        node_ids = set(seed_node_ids)
        
        # Queue for BFS traversal
        # Each entry is (node_id, current_depth)
        queue = [(node_id, 0) for node_id in seed_node_ids]
        # Set of visited nodes
        visited = set(seed_node_ids)
        
        while queue:
            node_id, depth = queue.pop(0)
            
            # Add node to subgraph
            node = self.get_node(node_id)
            if node:
                subgraph.add_node(node)
            
            # If reached max depth, don't traverse further
            if depth >= max_depth:
                continue
            
            # Process outgoing edges
            if node_id in self.outgoing:
                for target_id, relationship, properties in self.outgoing[node_id]:
                    # Add edge to subgraph
                    edge = KGEdge(
                        source_id=node_id,
                        target_id=target_id,
                        relationship=relationship,
                        properties=properties
                    )
                    target_node = self.get_node(target_id)
                    if target_node:
                        subgraph.add_node(target_node)
                        subgraph.add_edge(edge)
                    
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
                    edge = KGEdge(
                        source_id=source_id,
                        target_id=node_id,
                        relationship=relationship,
                        properties=properties
                    )
                    source_node = self.get_node(source_id)
                    if source_node:
                        subgraph.add_node(source_node)
                        subgraph.add_edge(edge)
                    
                    # Add source node to subgraph
                    node_ids.add(source_id)
                    
                    # Enqueue source node if not visited
                    if source_id not in visited:
                        visited.add(source_id)
                        queue.append((source_id, depth + 1))
        
        return subgraph
    
    def search_nodes(self, query: str) -> List[KGNode]:
        """
        Search for nodes by query string.
        
        Args:
            query: Search query
            
        Returns:
            List of matching nodes
        """
        query_lower = query.lower()
        results = []
        
        for node in self.nodes.values():
            # Check if query is in label
            if query_lower in node.label.lower():
                results.append(node)
                continue
            
            # Check if query is in properties
            for key, value in node.properties.items():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(node)
                    break
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert knowledge graph to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "nodes": [self._node_to_dict(node) for node in self.nodes.values()],
            "edges": [self._edge_to_dict(edge) for edge in self.edges]
        }
    
    @staticmethod
    def _node_to_dict(node: KGNode) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "id": node.id,
            "label": node.label,
            "properties": node.properties,
            "node_type": node.node_type
        }
    
    @staticmethod
    def _edge_to_dict(edge: KGEdge) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        return {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relationship": edge.relationship,
            "properties": edge.properties,
            "id": edge.id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeGraph':
        """
        Create knowledge graph from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Knowledge graph
        """
        graph = cls(name=data.get("name", "default"))
        
        # Add nodes
        for node_data in data.get("nodes", []):
            node = KGNode(
                id=node_data["id"],
                label=node_data["label"],
                properties=node_data.get("properties", {}),
                node_type=node_data.get("node_type", "entity")
            )
            graph.add_node(node)
        
        # Add edges
        for edge_data in data.get("edges", []):
            edge = KGEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                relationship=edge_data["relationship"],
                properties=edge_data.get("properties", {}),
                id=edge_data.get("id")
            )
            graph.add_edge(edge)
        
        return graph
    
    def to_json(self) -> str:
        """
        Convert knowledge graph to JSON.
        
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'KnowledgeGraph':
        """
        Create knowledge graph from JSON.
        
        Args:
            json_str: JSON string
            
        Returns:
            Knowledge graph
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class KGQueryEngine:
    """
    Query engine for knowledge graphs.
    Provides methods for entity extraction and retrieval-based queries.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, entity_extractor: Optional[Callable] = None):
        """
        Initialize the query engine.
        
        Args:
            knowledge_graph: Knowledge graph to query
            entity_extractor: Optional function to extract entities from text
        """
        self.knowledge_graph = knowledge_graph
        self.entity_extractor = entity_extractor
    
    def query(self, query_text: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Query the knowledge graph.
        
        Args:
            query_text: Query text
            max_depth: Maximum traversal depth
            
        Returns:
            Query results
        """
        # Extract entities from query
        entities = self._extract_entities(query_text)
        
        # Get entity nodes
        entity_nodes = []
        
        for entity in entities:
            # Try to find by ID
            node = self.knowledge_graph.get_node(entity)
            if node:
                entity_nodes.append(node)
                continue
            
            # Try to find by label
            nodes = self.knowledge_graph.get_nodes_by_label(entity)
            entity_nodes.extend(nodes)
        
        # Extract subgraph
        subgraph = self.knowledge_graph.get_subgraph(
            [node.id for node in entity_nodes],
            max_depth=max_depth
        )
        
        # Format results
        return {
            "query": query_text,
            "entities": entities,
            "entity_nodes": entity_nodes,
            "subgraph": subgraph.to_dict()
        }
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        # Use entity extractor if provided
        if self.entity_extractor:
            return self.entity_extractor(text)
        
        # Simple fallback: check for known node labels
        entities = []
        
        for label in self.knowledge_graph.label_to_nodes.keys():
            if label.lower() in text.lower():
                entities.append(label)
        
        return entities


class RAGKnowledgeGraphIntegration:
    """
    Integration of knowledge graphs with RAG systems.
    Enriches retrieval with structured knowledge.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, query_engine: Optional[KGQueryEngine] = None):
        """
        Initialize the integration.
        
        Args:
            knowledge_graph: Knowledge graph
            query_engine: Optional query engine (created if not provided)
        """
        self.knowledge_graph = knowledge_graph
        self.query_engine = query_engine or KGQueryEngine(knowledge_graph)
    
    def augment(self, query: str, retrieved_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Augment RAG with knowledge graph information.
        
        Args:
            query: User query
            retrieved_context: Optional retrieved context
            
        Returns:
            Augmented context with knowledge graph information
        """
        # Query knowledge graph
        kg_results = self.query_engine.query(query)
        
        # Find relevant entities in retrieved context if provided
        context_entities = []
        if retrieved_context:
            context_entities = self.query_engine._extract_entities(retrieved_context)
        
        # Get subgraph for context entities
        context_subgraph = None
        if context_entities:
            # Find nodes for context entities
            entity_nodes = []
            for entity in context_entities:
                # Try to find by ID
                node = self.knowledge_graph.get_node(entity)
                if node:
                    entity_nodes.append(node)
                    continue
                
                # Try to find by label
                nodes = self.knowledge_graph.get_nodes_by_label(entity)
                entity_nodes.extend(nodes)
            
            # Extract subgraph
            context_subgraph = self.knowledge_graph.get_subgraph(
                [node.id for node in entity_nodes],
                max_depth=1
            ).to_dict()
        
        # Format augmented context
        augmented_context = {
            "query": query,
            "kg_results": kg_results,
            "context_entities": context_entities,
            "context_subgraph": context_subgraph
        }
        
        return augmented_context
    
    def generate_kg_prompt(self, kg_results: Dict[str, Any]) -> str:
        """
        Generate prompt describing knowledge graph information.
        
        Args:
            kg_results: Results from query_engine.query()
            
        Returns:
            Prompt text describing the knowledge graph information
        """
        prompt = "Knowledge Graph Information:\n\n"
        
        # Add entity information
        if kg_results.get("entity_nodes"):
            prompt += "Entities:\n"
            for i, node in enumerate(kg_results["entity_nodes"]):
                prompt += f"- {node.label}"
                if node.properties.get("definition"):
                    prompt += f": {node.properties['definition']}"
                prompt += "\n"
            prompt += "\n"
        
        # Add relationships
        if kg_results.get("subgraph") and kg_results["subgraph"].get("edges"):
            prompt += "Relationships:\n"
            seen_relationships = set()
            
            for edge_data in kg_results["subgraph"]["edges"]:
                # Get source and target nodes
                source_id = edge_data["source_id"]
                target_id = edge_data["target_id"]
                relationship = edge_data["relationship"]
                
                # Find node labels
                source_label = self._find_node_label(kg_results["subgraph"]["nodes"], source_id)
                target_label = self._find_node_label(kg_results["subgraph"]["nodes"], target_id)
                
                if not source_label or not target_label:
                    continue
                
                # Generate relationship description
                rel_key = f"{source_id}_{relationship}_{target_id}"
                if rel_key not in seen_relationships:
                    prompt += f"- {source_label} {relationship} {target_label}\n"
                    seen_relationships.add(rel_key)
            
            prompt += "\n"
        
        return prompt
    
    @staticmethod
    def _find_node_label(nodes: List[Dict[str, Any]], node_id: str) -> Optional[str]:
        """Find node label by ID in a list of node dictionaries."""
        for node in nodes:
            if node["id"] == node_id:
                return node["label"]
        return None


# Example usage and utility functions

def create_ai_knowledge_graph() -> KnowledgeGraph:
    """Create an example AI knowledge graph."""
    kg = KnowledgeGraph("AI_Knowledge_Graph")
    
    # Add concept nodes
    concepts = [
        KGNode(id="ml", label="Machine Learning", 
              properties={"definition": "Field of AI that enables systems to learn from data"},
              node_type="concept"),
        KGNode(id="dl", label="Deep Learning", 
              properties={"definition": "Subset of ML using neural networks with multiple layers"},
              node_type="concept"),
        KGNode(id="nlp", label="Natural Language Processing", 
              properties={"definition": "Branch of AI dealing with human language"},
              node_type="concept"),
        KGNode(id="cv", label="Computer Vision", 
              properties={"definition": "Field of AI that enables machines to interpret visual information"},
              node_type="concept"),
        KGNode(id="rl", label="Reinforcement Learning", 
              properties={"definition": "ML paradigm where agents learn by interacting with environment"},
              node_type="concept"),
        KGNode(id="rag", label="Retrieval Augmented Generation", 
              properties={"definition": "Technique that enhances LLMs with external knowledge"},
              node_type="concept"),
    ]
    
    # Add model nodes
    models = [
        KGNode(id="bert", label="BERT", 
              properties={"full_name": "Bidirectional Encoder Representations from Transformers"},
              node_type="model"),
        KGNode(id="gpt", label="GPT", 
              properties={"full_name": "Generative Pre-trained Transformer"},
              node_type="model"),
        KGNode(id="llama", label="LLaMA", 
              properties={"full_name": "Large Language Model Meta AI"},
              node_type="model"),
        KGNode(id="t5", label="T5", 
              properties={"full_name": "Text-to-Text Transfer Transformer"},
              node_type="model"),
        KGNode(id="clip", label="CLIP", 
              properties={"full_name": "Contrastive Language-Image Pre-training"},
              node_type="model"),
    ]
    
    # Add technology nodes
    technologies = [
        KGNode(id="transformer", label="Transformer", 
              properties={"definition": "Neural network architecture using self-attention"},
              node_type="technology"),
        KGNode(id="cnn", label="CNN", 
              properties={"full_name": "Convolutional Neural Network"},
              node_type="technology"),
        KGNode(id="rnn", label="RNN", 
              properties={"full_name": "Recurrent Neural Network"},
              node_type="technology"),
        KGNode(id="llm", label="LLM", 
              properties={"full_name": "Large Language Model"},
              node_type="technology"),
        KGNode(id="vec_db", label="Vector Database", 
              properties={"definition": "Database optimized for similarity search of vectors"},
              node_type="technology"),
    ]
    
    # Add all nodes
    for node in concepts + models + technologies:
        kg.add_node(node)
    
    # Add edges
    edges = [
        # Concept hierarchies
        KGEdge(source_id="ml", target_id="dl", relationship="has_subset"),
        KGEdge(source_id="ml", target_id="rl", relationship="has_subset"),
        KGEdge(source_id="ml", target_id="nlp", relationship="has_application"),
        KGEdge(source_id="ml", target_id="cv", relationship="has_application"),
        
        # Technology relationships
        KGEdge(source_id="dl", target_id="transformer", relationship="uses_technology"),
        KGEdge(source_id="dl", target_id="cnn", relationship="uses_technology"),
        KGEdge(source_id="dl", target_id="rnn", relationship="uses_technology"),
        KGEdge(source_id="dl", target_id="llm", relationship="enables"),
        KGEdge(source_id="rag", target_id="vec_db", relationship="uses_technology"),
        KGEdge(source_id="rag", target_id="llm", relationship="augments"),
        
        # Model relationships
        KGEdge(source_id="transformer", target_id="bert", relationship="enables"),
        KGEdge(source_id="transformer", target_id="gpt", relationship="enables"),
        KGEdge(source_id="transformer", target_id="t5", relationship="enables"),
        KGEdge(source_id="transformer", target_id="llama", relationship="enables"),
        KGEdge(source_id="llm", target_id="gpt", relationship="has_example"),
        KGEdge(source_id="llm", target_id="llama", relationship="has_example"),
        KGEdge(source_id="nlp", target_id="bert", relationship="uses"),
        KGEdge(source_id="cv", target_id="clip", relationship="uses"),
        KGEdge(source_id="nlp", target_id="llm", relationship="uses"),
    ]
    
    # Add all edges
    for edge in edges:
        kg.add_edge(edge)
    
    return kg


def demonstrate_kg_integration():
    """Demonstrate knowledge graph integration with RAG."""
    print("Knowledge Graph Integration for RAG Systems")
    print("-------------------------------------------")
    
    # Create knowledge graph
    kg = create_ai_knowledge_graph()
    print(f"Created knowledge graph with {len(kg.nodes)} nodes and {len(kg.edges)} edges")
    
    # Create integration
    integration = RAGKnowledgeGraphIntegration(kg)
    
    # Example queries
    queries = [
        "How does BERT relate to transformers?",
        "What models use transformer architecture?",
        "How is RAG implemented with LLMs?",
        "What are the relationships between machine learning and NLP?",
    ]
    
    # Test each query
    for query in queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("-" * 80)
        
        # Augment with knowledge graph
        augmented = integration.augment(query)
        
        # Print entity information
        print("\nEntities found:")
        for entity in augmented["kg_results"]["entities"]:
            print(f"- {entity}")
        
        # Print knowledge graph prompt
        print("\nKnowledge Graph Prompt:")
        prompt = integration.generate_kg_prompt(augmented["kg_results"])
        print(prompt)
        
        print("=" * 80)


if __name__ == "__main__":
    demonstrate_kg_integration() 