#!/usr/bin/env python3
"""
MCP Server Implementation Guide
=============================

This module provides a complete implementation of a Multi-Command Protocol (MCP) server
that can connect to multiple services, route commands, and handle responses.

This implementation includes:
- Service registry management
- Command validation and routing
- Authentication and authorization
- Error handling and logging
- WebSocket support for real-time events
- Command history and tracking

Run with:
    python mcp_server_implementation.py

Dependencies:
    - fastapi
    - uvicorn
    - pydantic
    - websockets
    - python-jose (for JWT)
    - asyncio
"""

import os
import sys
import time
import json
import uuid
import logging
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Callable
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, validator

# Import command protocol definitions
# In your implementation, you would import from command_protocol.py
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# ===== Command Protocol Models (simplified) =====

class CommandStatus(str, Enum):
    """Status of a command during its lifecycle."""
    CREATED = "created"
    VALIDATED = "validated"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMEOUT = "timeout"

class ServiceStatus(str, Enum):
    """Status of a service connected to the MCP server."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"

class Command(BaseModel):
    """Model representing a command to be executed."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    service_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    status: CommandStatus = CommandStatus.CREATED
    user_id: Optional[str] = None
    timeout_seconds: int = 60

class CommandResult(BaseModel):
    """Result of a command execution."""
    command_id: str
    status: CommandStatus
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.now)

class Service(BaseModel):
    """Definition of a service connected to the MCP server."""
    id: str
    name: str
    url: str
    status: ServiceStatus = ServiceStatus.ONLINE
    description: str = ""
    capabilities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    health_check_url: Optional[str] = None

class User(BaseModel):
    """User definition with permissions."""
    id: str
    username: str
    permissions: List[str] = Field(default_factory=list)
    disabled: bool = False

# ===== MCP Server Implementation =====

class MCPServer:
    """
    Core MCP Server implementation that coordinates services and commands.
    
    This class manages:
    1. Service registration and discovery
    2. Command validation and routing
    3. Command execution and result handling
    4. Service health monitoring
    """
    
    def __init__(self):
        """Initialize the MCP server."""
        self.services: Dict[str, Service] = {}
        self.commands: Dict[str, Command] = {}
        self.results: Dict[str, CommandResult] = {}
        self.service_clients: Dict[str, Any] = {}
        self.active_websockets: Set[WebSocket] = set()
        self.command_history: List[Dict[str, Any]] = []
        
        # Configure logging
        self.logger = logging.getLogger("mcp_server")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Thread pool for handling blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Event loop for async operations
        self.event_loop = asyncio.get_event_loop()
        
        self.logger.info("MCP Server initialized")
    
    # ===== Service Registry Methods =====
    
    def register_service(self, service: Service) -> str:
        """Register a new service with the MCP server."""
        self.services[service.id] = service
        self.logger.info(f"Service registered: {service.name} ({service.id})")
        return service.id
    
    def deregister_service(self, service_id: str) -> bool:
        """Remove a service from the registry."""
        if service_id in self.services:
            service = self.services.pop(service_id)
            self.logger.info(f"Service deregistered: {service.name} ({service_id})")
            return True
        return False
    
    def get_service(self, service_id: str) -> Optional[Service]:
        """Get a service by ID."""
        return self.services.get(service_id)
    
    def list_services(self) -> List[Service]:
        """List all registered services."""
        return list(self.services.values())
    
    async def check_service_health(self, service_id: str) -> bool:
        """Check if a service is healthy."""
        service = self.get_service(service_id)
        if not service or not service.health_check_url:
            return False
        
        try:
            # In a real implementation, make an HTTP request to the health check URL
            # For this demo, we'll simulate a successful health check
            await asyncio.sleep(0.1)
            return True
        except Exception as e:
            self.logger.error(f"Health check failed for service {service_id}: {str(e)}")
            return False
    
    async def monitor_services_health(self, interval_seconds: int = 60):
        """Periodically monitor the health of all services."""
        while True:
            try:
                for service_id in list(self.services.keys()):
                    service = self.get_service(service_id)
                    if not service:
                        continue
                    
                    is_healthy = await self.check_service_health(service_id)
                    old_status = service.status
                    
                    if is_healthy and service.status != ServiceStatus.ONLINE:
                        service.status = ServiceStatus.ONLINE
                        self.logger.info(f"Service {service.name} is now ONLINE")
                    elif not is_healthy and service.status == ServiceStatus.ONLINE:
                        service.status = ServiceStatus.DEGRADED
                        self.logger.warning(f"Service {service.name} is DEGRADED")
                    
                    if old_status != service.status:
                        # Notify clients about status change
                        await self.broadcast_event({
                            "type": "service_status_change",
                            "service_id": service_id,
                            "old_status": old_status,
                            "new_status": service.status
                        })
            except Exception as e:
                self.logger.error(f"Error in service health monitoring: {str(e)}")
            
            await asyncio.sleep(interval_seconds)
    
    # ===== Command Handling Methods =====
    
    def validate_command(self, command: Command) -> Optional[str]:
        """
        Validate if a command can be executed.
        
        Returns None if validation passes, or error message if it fails.
        """
        # Check if target service exists
        service = self.get_service(command.service_id)
        if not service:
            return f"Service {command.service_id} not found"
        
        # Check if service is online
        if service.status == ServiceStatus.OFFLINE:
            return f"Service {command.service_id} is offline"
        
        # Check if service supports this command
        if command.name not in service.capabilities:
            return f"Service {command.service_id} does not support command {command.name}"
        
        # In a real implementation, check command parameters against a schema
        
        return None  # Validation passed
    
    async def execute_command(self, command: Command) -> CommandResult:
        """Execute a command by routing it to the appropriate service."""
        self.logger.info(f"Executing command: {command.id} ({command.name})")
        
        # Store the command
        self.commands[command.id] = command
        
        # Update command status
        command.status = CommandStatus.VALIDATED
        
        # Validate the command
        validation_error = self.validate_command(command)
        if validation_error:
            self.logger.error(f"Command validation failed: {validation_error}")
            result = CommandResult(
                command_id=command.id,
                status=CommandStatus.FAILED,
                error={"message": validation_error}
            )
            self.results[command.id] = result
            return result
        
        # Update command status
        command.status = CommandStatus.QUEUED
        
        # Get the target service
        service = self.get_service(command.service_id)
        
        try:
            # Update command status
            command.status = CommandStatus.IN_PROGRESS
            
            # Notify about status change
            await self.broadcast_event({
                "type": "command_status_change",
                "command_id": command.id,
                "status": command.status
            })
            
            # In a real implementation, make an HTTP request to the service
            # For this demo, we'll simulate a successful execution
            start_time = time.time()
            
            # Simulate different execution times based on command
            if "sleep" in command.parameters:
                await asyncio.sleep(float(command.parameters["sleep"]))
            else:
                await asyncio.sleep(0.5)
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Create a success result
            result = CommandResult(
                command_id=command.id,
                status=CommandStatus.COMPLETED,
                result=f"Command {command.name} executed successfully on {service.name}",
                execution_time_ms=execution_time_ms
            )
            
            # Update command status
            command.status = CommandStatus.COMPLETED
        except Exception as e:
            self.logger.error(f"Command execution failed: {str(e)}")
            result = CommandResult(
                command_id=command.id,
                status=CommandStatus.FAILED,
                error={"message": str(e)}
            )
            command.status = CommandStatus.FAILED
        
        # Store the result
        self.results[command.id] = result
        
        # Add to command history
        self.command_history.append({
            "command_id": command.id,
            "name": command.name,
            "service_id": command.service_id,
            "user_id": command.user_id,
            "status": command.status,
            "created_at": command.created_at.isoformat(),
            "execution_time_ms": result.execution_time_ms
        })
        
        # Notify about completion
        await self.broadcast_event({
            "type": "command_completed",
            "command_id": command.id,
            "status": command.status,
            "execution_time_ms": result.execution_time_ms
        })
        
        return result
    
    def get_command(self, command_id: str) -> Optional[Command]:
        """Get a command by ID."""
        return self.commands.get(command_id)
    
    def get_command_result(self, command_id: str) -> Optional[CommandResult]:
        """Get the result of a command."""
        return self.results.get(command_id)
    
    def list_commands(self, limit: int = 100, offset: int = 0) -> List[Command]:
        """List commands, optionally filtered and paginated."""
        return list(self.commands.values())[offset:offset+limit]
    
    # ===== WebSocket Methods =====
    
    async def register_websocket(self, websocket: WebSocket):
        """Register a WebSocket connection for real-time updates."""
        await websocket.accept()
        self.active_websockets.add(websocket)
        self.logger.info(f"WebSocket client connected: {websocket.client}")
    
    def deregister_websocket(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_websockets.discard(websocket)
        self.logger.info(f"WebSocket client disconnected: {websocket.client}")
    
    async def broadcast_event(self, event: Dict[str, Any]):
        """Broadcast an event to all connected WebSocket clients."""
        disconnected_websockets = set()
        
        for websocket in self.active_websockets:
            try:
                await websocket.send_json(event)
            except Exception:
                # Mark for removal
                disconnected_websockets.add(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected_websockets:
            self.deregister_websocket(websocket)
    
    # ===== Server Control Methods =====
    
    async def start(self):
        """Start the MCP server and background tasks."""
        # Start service health monitoring
        asyncio.create_task(self.monitor_services_health())
        self.logger.info("MCP Server started")
    
    async def stop(self):
        """Stop the MCP server gracefully."""
        # Close all WebSocket connections
        for websocket in self.active_websockets:
            try:
                await websocket.close()
            except:
                pass
        self.active_websockets.clear()
        
        # Shutdown thread pool
        self.thread_pool.shutdown()
        
        self.logger.info("MCP Server stopped")

# ===== FastAPI Integration =====

class MCPServerAPI:
    """
    FastAPI wrapper for the MCP Server.
    
    This class provides a REST API and WebSocket interface for interacting with
    the MCP server.
    """
    
    def __init__(self, mcp_server: MCPServer):
        """Initialize the API with an MCP server instance."""
        self.mcp_server = mcp_server
        self.app = FastAPI(title="MCP Server API", version="1.0.0")
        self.setup_routes()
        self.setup_middleware()
        
        # Authentication
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        
        # Mock user database (in a real app, use a proper database)
        self.users = {
            "user1": User(
                id="user1",
                username="demo_user",
                permissions=["commands:execute:*", "services:list"],
                disabled=False
            ),
            "admin": User(
                id="admin",
                username="admin_user",
                permissions=["*"],
                disabled=False
            )
        }
    
    def setup_middleware(self):
        """Set up middleware for the API."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Set up API routes."""
        app = self.app
        
        @app.on_event("startup")
        async def startup_event():
            """Start the MCP server on API startup."""
            await self.mcp_server.start()
        
        @app.on_event("shutdown")
        async def shutdown_event():
            """Stop the MCP server on API shutdown."""
            await self.mcp_server.stop()
        
        @app.get("/")
        async def root():
            """Root endpoint."""
            return {"message": "MCP Server API", "version": "1.0.0"}
        
        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # ===== Service Endpoints =====
        
        @app.post("/services")
        async def register_service(
            service: Service,
            current_user: User = Depends(self.get_current_user)
        ):
            """Register a new service."""
            self.check_permission(current_user, "services:register")
            service_id = self.mcp_server.register_service(service)
            return {"service_id": service_id}
        
        @app.delete("/services/{service_id}")
        async def deregister_service(
            service_id: str,
            current_user: User = Depends(self.get_current_user)
        ):
            """Deregister a service."""
            self.check_permission(current_user, "services:deregister")
            success = self.mcp_server.deregister_service(service_id)
            if not success:
                raise HTTPException(status_code=404, detail="Service not found")
            return {"success": True}
        
        @app.get("/services")
        async def list_services(
            current_user: User = Depends(self.get_current_user)
        ):
            """List all services."""
            self.check_permission(current_user, "services:list")
            services = self.mcp_server.list_services()
            return {"services": [s.dict() for s in services]}
        
        @app.get("/services/{service_id}")
        async def get_service(
            service_id: str,
            current_user: User = Depends(self.get_current_user)
        ):
            """Get a service by ID."""
            self.check_permission(current_user, "services:get")
            service = self.mcp_server.get_service(service_id)
            if not service:
                raise HTTPException(status_code=404, detail="Service not found")
            return service
        
        # ===== Command Endpoints =====
        
        @app.post("/commands")
        async def execute_command(
            command: Command,
            current_user: User = Depends(self.get_current_user)
        ):
            """Execute a new command."""
            permission = f"commands:execute:{command.name}"
            self.check_permission(current_user, permission)
            
            # Set user ID
            command.user_id = current_user.id
            
            # Execute the command
            result = await self.mcp_server.execute_command(command)
            
            return result
        
        @app.get("/commands")
        async def list_commands(
            limit: int = 100,
            offset: int = 0,
            current_user: User = Depends(self.get_current_user)
        ):
            """List commands."""
            self.check_permission(current_user, "commands:list")
            commands = self.mcp_server.list_commands(limit, offset)
            return {"commands": [c.dict() for c in commands]}
        
        @app.get("/commands/{command_id}")
        async def get_command(
            command_id: str,
            current_user: User = Depends(self.get_current_user)
        ):
            """Get a command by ID."""
            self.check_permission(current_user, "commands:get")
            command = self.mcp_server.get_command(command_id)
            if not command:
                raise HTTPException(status_code=404, detail="Command not found")
            return command
        
        @app.get("/commands/{command_id}/result")
        async def get_command_result(
            command_id: str,
            current_user: User = Depends(self.get_current_user)
        ):
            """Get the result of a command."""
            self.check_permission(current_user, "commands:get")
            result = self.mcp_server.get_command_result(command_id)
            if not result:
                raise HTTPException(status_code=404, detail="Command result not found")
            return result
        
        # ===== WebSocket Endpoint =====
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.mcp_server.register_websocket(websocket)
            try:
                while True:
                    # Wait for messages (not used in this demo)
                    data = await websocket.receive_text()
            except WebSocketDisconnect:
                self.mcp_server.deregister_websocket(websocket)
    
    async def get_current_user(self, token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))):
        """Get the current user from JWT token."""
        # In a real implementation, validate JWT token
        # For this demo, we just check if the token exists in our mock user database
        
        if token not in self.users:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = self.users[token]
        if user.disabled:
            raise HTTPException(status_code=400, detail="Inactive user")
        
        return user
    
    def check_permission(self, user: User, required_permission: str):
        """Check if a user has the required permission."""
        # Admin can do anything
        if "*" in user.permissions:
            return True
        
        # Check for specific permission
        if required_permission in user.permissions:
            return True
        
        # Check for wildcard permissions
        for permission in user.permissions:
            if permission.endswith(":*"):
                prefix = permission[:-1]
                if required_permission.startswith(prefix):
                    return True
        
        # Permission denied
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {required_permission}"
        )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server."""
        uvicorn.run(self.app, host=host, port=port)

# ===== Example MCP Server Setup =====

def create_example_mcp_server():
    """Create and configure an example MCP server."""
    # Create the core MCP server
    mcp_server = MCPServer()
    
    # Register some example services
    mcp_server.register_service(Service(
        id="llm-service-1",
        name="GPT-4 Turbo Service",
        url="https://api.example.com/llm",
        capabilities=["generate_text", "analyze_sentiment", "summarize"],
        description="Provides access to GPT-4 Turbo for text generation and analysis",
        health_check_url="https://api.example.com/llm/health"
    ))
    
    mcp_server.register_service(Service(
        id="tool-service-1",
        name="Image Processing Service",
        url="https://api.example.com/tools/image",
        capabilities=["resize_image", "filter_image", "recognize_objects"],
        description="Provides image processing and analysis capabilities",
        health_check_url="https://api.example.com/tools/image/health"
    ))
    
    mcp_server.register_service(Service(
        id="db-service-1",
        name="Vector Database Service",
        url="https://api.example.com/vector-db",
        capabilities=["store_vectors", "query_vectors", "delete_vectors"],
        description="Provides vector database operations for semantic search",
        health_check_url="https://api.example.com/vector-db/health"
    ))
    
    return mcp_server

# ===== Client Usage Examples =====

class MCPClient:
    """
    Example client for interacting with an MCP server.
    
    This is a simplified client for demonstration purposes.
    In a real implementation, handle authentication, retries, etc.
    """
    
    def __init__(self, api_url: str, auth_token: str):
        """Initialize the client."""
        self.api_url = api_url
        self.auth_headers = {"Authorization": f"Bearer {auth_token}"}
    
    async def list_services(self):
        """List available services."""
        print("Listing services...")
        # In a real client, make HTTP request to /services
        await asyncio.sleep(0.1)
        return [
            {"id": "llm-service-1", "name": "GPT-4 Turbo Service"},
            {"id": "tool-service-1", "name": "Image Processing Service"},
            {"id": "db-service-1", "name": "Vector Database Service"}
        ]
    
    async def execute_command(self, name: str, service_id: str, parameters: Dict[str, Any]):
        """Execute a command on the MCP server."""
        print(f"Executing command: {name} on service {service_id}")
        # In a real client, make HTTP POST request to /commands
        command_id = str(uuid.uuid4())
        await asyncio.sleep(0.5)
        
        # Simulate response
        return {
            "command_id": command_id,
            "status": "completed",
            "result": f"Simulated execution of {name} on {service_id}",
            "execution_time_ms": 500
        }
    
    async def get_command_result(self, command_id: str):
        """Get the result of a command."""
        print(f"Getting result for command: {command_id}")
        # In a real client, make HTTP GET request to /commands/{command_id}/result
        await asyncio.sleep(0.1)
        
        # Simulate response
        return {
            "command_id": command_id,
            "status": "completed",
            "result": "Command executed successfully",
            "execution_time_ms": 500
        }
    
    async def connect_websocket(self):
        """Connect to the WebSocket for real-time updates."""
        print("Connecting to WebSocket...")
        # In a real client, establish WebSocket connection
        
        # Simulate connection
        await asyncio.sleep(0.1)
        print("WebSocket connected")
        
        # In a real client, start receiving messages
        async def receive_messages():
            while True:
                # Simulate receiving a message
                await asyncio.sleep(2)
                yield {
                    "type": "command_completed",
                    "command_id": str(uuid.uuid4()),
                    "status": "completed"
                }
        
        return receive_messages()

# ===== Example Usage =====

async def client_example():
    """Example usage of the MCP client."""
    client = MCPClient(api_url="http://localhost:8000", auth_token="user1")
    
    # List services
    services = await client.list_services()
    print(f"Available services: {[s['name'] for s in services]}")
    
    # Execute a command
    result = await client.execute_command(
        name="generate_text",
        service_id="llm-service-1",
        parameters={
            "prompt": "Explain MCP servers in simple terms",
            "max_tokens": 100
        }
    )
    print(f"Command execution result: {result}")
    
    # Get real-time updates
    messages = await client.connect_websocket()
    try:
        # Get a few messages then stop
        counter = 0
        async for message in messages:
            print(f"WebSocket message: {message}")
            counter += 1
            if counter >= 3:
                break
    except Exception as e:
        print(f"WebSocket error: {str(e)}")

# ===== Main Function =====

async def main_async():
    """Asynchronous main function."""
    print("MCP Server Implementation Example")
    print("=================================")
    
    # Create and start MCP server
    mcp_server = create_example_mcp_server()
    await mcp_server.start()
    
    print("\nServer started with services:")
    for service in mcp_server.list_services():
        print(f"- {service.name} ({service.id}): {len(service.capabilities)} capabilities")
    
    # Execute a sample command
    print("\nExecuting sample command...")
    command = Command(
        name="generate_text",
        service_id="llm-service-1",
        parameters={
            "prompt": "Explain MCP servers",
            "max_tokens": 100
        },
        user_id="demo-user"
    )
    
    result = await mcp_server.execute_command(command)
    print(f"Command result: {result.dict()}")
    
    # Show client example
    print("\nClient example:")
    await client_example()
    
    # Clean up
    await mcp_server.stop()
    print("\nServer stopped")

def main():
    """Main function."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 