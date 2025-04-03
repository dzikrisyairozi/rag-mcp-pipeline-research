#!/usr/bin/env python3
"""
MCP Architecture Visualization and Explanation
=============================================

This script provides an interactive visualization and explanation of
Multi-Context Protocol (MCP) server architecture, including components,
communication flows, and design patterns.

Run with:
    python mcp_architecture.py

Dependencies:
    - gradio
    - matplotlib
    - networkx
    - pydantic
"""

import os
import sys
import json
import time
import gradio as gr
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

# ===== MCP Component Models =====

class ServiceStatus(str, Enum):
    """Status of a service connected to the MCP server."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"

class ServiceType(str, Enum):
    """Types of services that can connect to an MCP server."""
    LLM = "llm"
    TOOL = "tool"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    FILE_STORAGE = "file_storage"
    CUSTOM = "custom"

class ContextStatus(str, Enum):
    """Status of a context being processed by the MCP server."""
    RECEIVED = "received"
    VALIDATING = "validating"
    ROUTING = "routing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

class Context(BaseModel):
    """Represents a standardized context in the MCP ecosystem."""
    id: str
    name: str
    service_target: str
    parameters: Dict[str, Any]
    require_auth: bool = True
    timeout_seconds: int = 30
    status: ContextStatus = ContextStatus.RECEIVED
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "service_target": self.service_target,
            "parameters": self.parameters,
            "require_auth": self.require_auth,
            "timeout_seconds": self.timeout_seconds,
            "status": self.status
        }

class Service(BaseModel):
    """Represents a service connected to the MCP server."""
    id: str
    name: str
    type: ServiceType
    status: ServiceStatus = ServiceStatus.ONLINE
    description: str = ""
    endpoint: str
    capabilities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "status": self.status,
            "description": self.description,
            "endpoint": self.endpoint,
            "capabilities": self.capabilities,
            "metadata": self.metadata
        }

class Client(BaseModel):
    """Represents a client connecting to the MCP server."""
    id: str
    name: str
    api_key: str
    permissions: List[str] = Field(default_factory=list)
    rate_limit: int = 100  # requests per minute
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "permissions": self.permissions,
            "rate_limit": self.rate_limit
        }

@dataclass
class MCPServer:
    """Represents the core MCP server that coordinates contexts."""
    id: str
    name: str
    services: Dict[str, Service] = field(default_factory=dict)
    clients: Dict[str, Client] = field(default_factory=dict)
    active_contexts: Dict[str, Context] = field(default_factory=dict)
    
    def register_service(self, service: Service):
        """Register a new service with the MCP server."""
        self.services[service.id] = service
        return f"Service {service.name} registered successfully"
    
    def deregister_service(self, service_id: str):
        """Remove a service from the MCP server."""
        if service_id in self.services:
            service = self.services.pop(service_id)
            return f"Service {service.name} deregistered successfully"
        return f"Service {service_id} not found"
    
    def register_client(self, client: Client):
        """Register a new client with the MCP server."""
        self.clients[client.id] = client
        return f"Client {client.name} registered successfully"
    
    def validate_context(self, context: Context) -> bool:
        """Validate if a context can be executed."""
        # Check if target service exists
        if context.service_target not in self.services:
            return False
        
        # Check if service is online
        if self.services[context.service_target].status != ServiceStatus.ONLINE:
            return False
            
        # Check if service can handle this context
        if context.name not in self.services[context.service_target].capabilities:
            return False
            
        return True
    
    def route_context(self, context: Context) -> Optional[str]:
        """Route a context to the appropriate service."""
        if not self.validate_context(context):
            return None
        
        # In a real system, this would communicate with the service
        # For this demo, we'll just return the endpoint
        context.status = ContextStatus.ROUTING
        return self.services[context.service_target].endpoint
    
    def execute_context(self, context: Context, client_id: str) -> Dict[str, Any]:
        """Execute a context on behalf of a client."""
        # Validate client permissions (simplified)
        if client_id not in self.clients:
            return {"error": "Client not authorized"}
        
        # Update context status
        context.status = ContextStatus.VALIDATING
        
        # Validate and route context
        endpoint = self.route_context(context)
        if not endpoint:
            context.status = ContextStatus.FAILED
            return {"error": "Context routing failed"}
        
        # In a real system, this would make the actual API call
        # For this demo, we'll simulate success
        context.status = ContextStatus.EXECUTING
        time.sleep(0.5)  # Simulate processing time
        
        # Track the context
        self.active_contexts[context.id] = context
        
        # Update context status
        context.status = ContextStatus.COMPLETED
        
        return {
            "context_id": context.id,
            "status": context.status,
            "result": f"Context {context.name} executed successfully"
        }

    def to_dict(self):
        """Convert the MCP server to a dictionary for visualization."""
        return {
            "id": self.id,
            "name": self.name,
            "services": {k: v.to_dict() for k, v in self.services.items()},
            "clients": {k: v.to_dict() for k, v in self.clients.items()},
            "active_contexts": {k: v.to_dict() for k, v in self.active_contexts.items()}
        }

# ===== Visualization Functions =====

def visualize_mcp_components():
    """Create a visualization of MCP components and their relationships."""
    G = nx.DiGraph()
    
    # Add nodes for each component type
    G.add_node("Client", pos=(0, 0), node_color='skyblue')
    G.add_node("API Gateway", pos=(1, 0), node_color='lightgreen')
    G.add_node("Auth Service", pos=(1, -1), node_color='lightgreen')
    G.add_node("MCP Server", pos=(2, 0), node_color='gold')
    G.add_node("Service Registry", pos=(2, -1), node_color='gold')
    G.add_node("Command Router", pos=(2, 1), node_color='gold')
    G.add_node("LLM Service", pos=(3, 1), node_color='salmon')
    G.add_node("Tool Service", pos=(3, 0), node_color='salmon')
    G.add_node("Database", pos=(3, -1), node_color='salmon')
    
    # Add edges to show relationships
    edges = [
        ("Client", "API Gateway", "Commands"),
        ("API Gateway", "Auth Service", "Verify"),
        ("API Gateway", "MCP Server", "Forward"),
        ("MCP Server", "Service Registry", "Lookup"),
        ("MCP Server", "Command Router", "Route"),
        ("Command Router", "LLM Service", "Execute"),
        ("Command Router", "Tool Service", "Execute"),
        ("Command Router", "Database", "Query"),
        ("LLM Service", "Command Router", "Response"),
        ("Tool Service", "Command Router", "Response"),
        ("Database", "Command Router", "Results"),
        ("Command Router", "MCP Server", "Return"),
        ("MCP Server", "API Gateway", "Response"),
        ("API Gateway", "Client", "Results")
    ]
    
    # Add edges with labels
    for src, dst, label in edges:
        G.add_edge(src, dst, label=label)
    
    # Set up plot
    plt.figure(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes
    node_colors = [G.nodes[n].get('node_color', 'lightblue') for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='gray', 
                          connectionstyle='arc3,rad=0.1', arrowsize=15)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10,
                                label_pos=0.3, bbox=dict(alpha=0))
    
    # Save figure
    plt.axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "mcp_architecture.png")
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def visualize_command_flow(mcp_server: MCPServer, context: Context):
    """Visualize the flow of a context through the MCP system."""
    G = nx.DiGraph()
    
    # Add nodes for the flow
    G.add_node("Client", pos=(0, 0), node_color='skyblue')
    G.add_node("API Gateway", pos=(1, 0), node_color='lightgreen')
    G.add_node("MCP Server", pos=(2, 0), node_color='gold')
    G.add_node(f"Service: {context.service_target}", pos=(3, 0), 
               node_color='salmon')
    
    # Add edges to show flow
    edges = [
        ("Client", "API Gateway", "1. Send Context"),
        ("API Gateway", "MCP Server", "2. Validate & Route"),
        ("MCP Server", f"Service: {context.service_target}", "3. Execute"),
        (f"Service: {context.service_target}", "MCP Server", "4. Return Result"),
        ("MCP Server", "API Gateway", "5. Process Response"),
        ("API Gateway", "Client", "6. Deliver Result")
    ]
    
    # Add edges with labels
    for src, dst, label in edges:
        G.add_edge(src, dst, label=label)
    
    # Set up plot
    plt.figure(figsize=(12, 4))
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes
    node_colors = [G.nodes[n].get('node_color', 'lightblue') for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='gray', 
                          arrowsize=15)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10,
                                bbox=dict(alpha=0))
    
    # Add context details as text
    cmd_details = f"Context: {context.name}\nParameters: {json.dumps(context.parameters, indent=2)}"
    plt.figtext(0.5, 0.01, cmd_details, ha="center", fontsize=10, 
               bbox={"boxstyle": "round", "alpha": 0.1})
    
    # Save figure
    plt.axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "command_flow.png")
    plt.savefig(output_path)
    plt.close()
    
    return output_path

# ===== Demo Functions =====

def create_demo_mcp_server():
    """Create a demo MCP server with sample services and clients."""
    # Create MCP server
    mcp = MCPServer(id="mcp-demo-1", name="Demo MCP Server")
    
    # Add sample services
    llm_service = Service(
        id="llm-service-1",
        name="GPT-4 Turbo Service",
        type=ServiceType.LLM,
        endpoint="https://api.example.com/llm",
        capabilities=["generate_text", "analyze_sentiment", "summarize"],
        description="Provides access to GPT-4 Turbo for text generation and analysis"
    )
    
    tool_service = Service(
        id="tool-service-1",
        name="Image Processing Service",
        type=ServiceType.TOOL,
        endpoint="https://api.example.com/tools/image",
        capabilities=["resize_image", "filter_image", "recognize_objects"],
        description="Provides image processing and analysis capabilities"
    )
    
    db_service = Service(
        id="db-service-1",
        name="Vector Database Service",
        type=ServiceType.DATABASE,
        endpoint="https://api.example.com/vector-db",
        capabilities=["store_vectors", "query_vectors", "delete_vectors"],
        description="Provides vector database operations for semantic search"
    )
    
    # Register services
    mcp.register_service(llm_service)
    mcp.register_service(tool_service)
    mcp.register_service(db_service)
    
    # Add sample clients
    client = Client(
        id="client-1",
        name="Web Application",
        api_key="api-key-12345",
        permissions=["llm-service-1:*", "tool-service-1:*", "db-service-1:query_vectors"]
    )
    
    # Register client
    mcp.register_client(client)
    
    return mcp

def create_demo_context():
    """Create a sample context for demonstration."""
    return Context(
        id="ctx-12345",
        name="generate_text",
        service_target="llm-service-1",
        parameters={
            "prompt": "Explain MCP architecture in simple terms",
            "max_tokens": 150,
            "temperature": 0.7
        }
    )

# ===== Gradio Interface =====

def render_mcp_explanation():
    """Render an explanation of MCP architecture."""
    return """
    # Multi-Context Protocol (MCP) Server Architecture
    
    An MCP server acts as a central coordinator in AI systems, handling:
    
    1. **Context Standardization**: Ensures all contexts follow a consistent format
    2. **Service Discovery**: Maintains a registry of available services
    3. **Context Routing**: Directs contexts to appropriate services
    4. **Authentication & Authorization**: Verifies clients and permissions
    5. **Error Handling**: Manages failures and provides consistent responses
    
    ## Key Components
    
    - **API Gateway**: Entry point for client requests, handles validation and rate limiting
    - **Service Registry**: Maintains information about available services
    - **Command Router**: Determines which service should handle each context
    - **Auth Service**: Manages authentication and authorization
    - **Services**: External functionality providers (LLMs, tools, databases, etc.)
    
    ## Design Benefits
    
    - **Decoupling**: Clients don't need to know service implementation details
    - **Consistency**: Standardized context format across all services
    - **Scalability**: New services can be added without client changes
    - **Security**: Centralized authentication and authorization
    - **Observability**: Central logging and monitoring of all contexts
    """

def visualize_full_architecture():
    """Generate and display the full MCP architecture visualization."""
    return visualize_mcp_components()

def execute_demo_context(context_name, service_target, parameters):
    """Execute a demo context and visualize its flow."""
    # Create MCP server with demo services
    mcp = create_demo_mcp_server()
    
    # Parse parameters from string input
    try:
        param_dict = json.loads(parameters)
    except:
        param_dict = {"error": "Invalid JSON parameters"}
    
    # Create context
    context = Context(
        id=f"ctx-{int(time.time())}",
        name=context_name,
        service_target=service_target,
        parameters=param_dict
    )
    
    # Execute context
    result = mcp.execute_context(context, "client-1")
    
    # Generate visualization
    flow_image = visualize_command_flow(mcp, context)
    
    # Return results
    return flow_image, json.dumps(result, indent=2)

def update_service_selection(service_type):
    """Update available service targets based on service type."""
    if service_type == ServiceType.LLM:
        return ["llm-service-1"], "generate_text"
    elif service_type == ServiceType.TOOL:
        return ["tool-service-1"], "resize_image"
    elif service_type == ServiceType.DATABASE:
        return ["db-service-1"], "query_vectors"
    else:
        return [], ""

# ===== Main Function =====

def main():
    """Create the Gradio interface for MCP architecture visualization."""
    with gr.Blocks(title="MCP Architecture Explorer") as app:
        gr.Markdown("# MCP Server Architecture Explorer")
        
        with gr.Tab("Architecture Overview"):
            gr.Markdown(render_mcp_explanation())
            architecture_btn = gr.Button("Visualize Architecture")
            architecture_img = gr.Image(type="filepath", label="MCP Architecture")
            architecture_btn.click(visualize_full_architecture, 
                                   inputs=[], 
                                   outputs=[architecture_img])
        
        with gr.Tab("Context Flow Simulation"):
            gr.Markdown("## Simulate Context Execution Flow")
            with gr.Row():
                with gr.Column():
                    service_type = gr.Dropdown(
                        choices=[x.value for x in ServiceType],
                        label="Service Type",
                        value=ServiceType.LLM
                    )
                    service_target = gr.Dropdown(
                        choices=["llm-service-1"],
                        label="Service Target",
                        value="llm-service-1"
                    )
                    context_name = gr.Dropdown(
                        choices=["generate_text"],
                        label="Context Name",
                        value="generate_text"
                    )
                    parameters = gr.Textbox(
                        label="Context Parameters (JSON)",
                        value="""{"prompt": "Explain MCP architecture", "max_tokens": 150}"""
                    )
                    execute_btn = gr.Button("Execute Context")
                
                with gr.Column():
                    flow_img = gr.Image(type="filepath", label="Context Flow")
                    result_json = gr.JSON(label="Execution Result")
            
            # Update available services based on type
            service_type.change(
                update_service_selection,
                inputs=[service_type],
                outputs=[service_target, context_name]
            )
            
            # Execute context button
            execute_btn.click(
                execute_demo_context,
                inputs=[context_name, service_target, parameters],
                outputs=[flow_img, result_json]
            )
    
    # Launch the interface
    print(f"MCP Architecture Explorer is starting... Open your browser to interact.")
    app.launch(share=False)

if __name__ == "__main__":
    # Generate architecture diagram on startup
    visualize_mcp_components()
    
    # Create a demo context flow visualization
    mcp = create_demo_mcp_server()
    ctx = create_demo_context()
    visualize_command_flow(mcp, ctx)
    
    print(f"MCP architecture diagrams created in {output_dir}")
    print("Run this script directly to launch the interactive explorer")
    
    # Uncomment to launch the Gradio interface immediately
    # main() 