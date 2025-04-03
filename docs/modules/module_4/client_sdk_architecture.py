#!/usr/bin/env python3
"""
MCP Client SDK Architecture
===========================

This module demonstrates the architecture for a client SDK that interacts with
MCP servers. It includes transport abstraction, authentication handling,
request/response models, and configuration management.

Key components:
- BaseClient: Abstract client with core functionality
- Transport implementations (HTTP, WebSocket)
- Authentication managers
- Configuration handling
- Request/Response models
"""

import os
import sys
import json
import time
import logging
import requests
import websocket
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_client")

# Type definitions
T = TypeVar('T')
Context = Dict[str, Any]
ContextResult = Dict[str, Any]

# ===== Models =====

class ContextStatus(Enum):
    """Enum representing the possible states of a context."""
    CREATED = "created"
    VALIDATED = "validated"
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class ClientConfig:
    """Configuration settings for the MCP client."""
    server_url: str
    api_version: str = "v1"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff: float = 1.5
    enable_websocket: bool = False
    websocket_url: Optional[str] = None
    verify_ssl: bool = True
    user_agent: str = "MCP-Client-SDK/1.0"
    
    def __post_init__(self):
        """Validate and set derived properties."""
        if not self.server_url:
            raise ValueError("server_url must be provided")
        
        # Set WebSocket URL if not provided but enabled
        if self.enable_websocket and not self.websocket_url:
            if self.server_url.startswith("https"):
                self.websocket_url = self.server_url.replace("https", "wss")
            else:
                self.websocket_url = self.server_url.replace("http", "ws")
            
            if not self.websocket_url.endswith("/ws"):
                self.websocket_url += "/ws"


@dataclass
class ContextRequest:
    """Request model for creating a new context."""
    context_name: str
    service_target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    priority: str = "normal"
    idempotency_key: Optional[str] = None
    
    def validate(self) -> None:
        """Validate the request is well-formed."""
        if not self.context_name:
            raise ValueError("context_name must be provided")
        if not self.service_target:
            raise ValueError("service_target must be provided")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")


@dataclass
class ContextResponse:
    """Response model for context execution."""
    context_id: str
    status: ContextStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ===== Authentication Providers =====

class AuthProvider(ABC):
    """Abstract base class for authentication providers."""
    
    @abstractmethod
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests."""
        pass
    
    @abstractmethod
    def refresh_if_needed(self) -> bool:
        """Refresh authentication if needed."""
        pass


class ApiKeyAuthProvider(AuthProvider):
    """API Key based authentication provider."""
    
    def __init__(self, api_key: str, header_name: str = "X-API-Key"):
        self.api_key = api_key
        self.header_name = header_name
    
    def get_auth_headers(self) -> Dict[str, str]:
        return {self.header_name: self.api_key}
    
    def refresh_if_needed(self) -> bool:
        # API keys don't need refreshing
        return True


class JwtAuthProvider(AuthProvider):
    """JWT based authentication provider with refresh capability."""
    
    def __init__(
        self, 
        token_url: str, 
        client_id: str, 
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
        token: Optional[str] = None,
        token_expiry: Optional[datetime] = None
    ):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.token = token
        self.token_expiry = token_expiry
    
    def get_auth_headers(self) -> Dict[str, str]:
        self.refresh_if_needed()
        if not self.token:
            raise ValueError("No valid token available")
        return {"Authorization": f"Bearer {self.token}"}
    
    def refresh_if_needed(self) -> bool:
        # If token is still valid or no refresh_token, return early
        if self.token and self.token_expiry and self.token_expiry > datetime.now() + timedelta(minutes=5):
            return True
        
        if not self.refresh_token and not self.client_secret:
            logger.error("Unable to refresh token: No refresh token or client credentials available")
            return False
        
        try:
            # Prepare refresh request
            if self.refresh_token:
                payload = {
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": self.client_id
                }
                if self.client_secret:
                    payload["client_secret"] = self.client_secret
            else:
                payload = {
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                }
            
            # Make request
            response = requests.post(self.token_url, data=payload)
            response.raise_for_status()
            token_data = response.json()
            
            # Update tokens
            self.token = token_data.get("access_token")
            if "refresh_token" in token_data:
                self.refresh_token = token_data.get("refresh_token")
            
            # Set expiry (default to 1 hour if not provided)
            expires_in = token_data.get("expires_in", 3600)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
            
            return True
            
        except Exception as e:
            logger.error(f"Token refresh failed: {str(e)}")
            return False


# ===== Transport Implementations =====

class Transport(ABC):
    """Abstract base class for transport implementations."""
    
    @abstractmethod
    def execute_context(self, request: ContextRequest) -> ContextResponse:
        """Execute a context and return the response."""
        pass
    
    @abstractmethod
    def get_context_status(self, context_id: str) -> ContextResponse:
        """Get the status of a context execution."""
        pass


class HttpTransport(Transport):
    """HTTP-based transport implementation."""
    
    def __init__(self, config: ClientConfig, auth_provider: Optional[AuthProvider] = None):
        self.config = config
        self.auth_provider = auth_provider
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.user_agent,
            "Content-Type": "application/json"
        })
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for the request, including auth if available."""
        headers = {}
        if self.auth_provider:
            headers.update(self.auth_provider.get_auth_headers())
        return headers
    
    def execute_context(self, request: ContextRequest) -> ContextResponse:
        """Execute a context via HTTP POST."""
        request.validate()
        
        # Attempt auth refresh if provider exists
        if self.auth_provider:
            self.auth_provider.refresh_if_needed()
        
        # Create JSON payload
        payload = {
            "context_name": request.context_name,
            "service_target": request.service_target,
            "parameters": request.parameters,
            "timeout_seconds": request.timeout_seconds,
            "priority": request.priority
        }
        
        if request.idempotency_key:
            payload["idempotency_key"] = request.idempotency_key
        
        headers = self._get_headers()
        if request.idempotency_key:
            headers["X-Idempotency-Key"] = request.idempotency_key
        
        # Prepare for retries
        retries_left = self.config.max_retries
        retry_delay = 1.0
        
        while retries_left >= 0:
            try:
                endpoint = f"{self.config.server_url}/api/{self.config.api_version}/context"
                response = self.session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=request.timeout_seconds,
                    verify=self.config.verify_ssl
                )
                
                # Handle response
                response.raise_for_status()
                data = response.json()
                
                # Parse result
                return ContextResponse(
                    context_id=data.get("context_id"),
                    status=ContextStatus(data.get("status", "created")),
                    result=data.get("result"),
                    error=data.get("error"),
                    execution_time_ms=data.get("execution_time_ms"),
                    created_at=datetime.fromisoformat(data.get("created_at")) if data.get("created_at") else None,
                    completed_at=datetime.fromisoformat(data.get("completed_at")) if data.get("completed_at") else None
                )
                
            except (requests.RequestException, ValueError) as e:
                retries_left -= 1
                if retries_left < 0:
                    logger.error(f"Failed to execute context after all retries: {str(e)}")
                    raise
                
                logger.warning(f"Request failed, retrying ({self.config.max_retries - retries_left}/{self.config.max_retries}): {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= self.config.retry_backoff
    
    def get_context_status(self, context_id: str) -> ContextResponse:
        """Get the status of a context execution."""
        if not context_id:
            raise ValueError("context_id must be provided")
        
        # Attempt auth refresh if provider exists
        if self.auth_provider:
            self.auth_provider.refresh_if_needed()
        
        endpoint = f"{self.config.server_url}/api/{self.config.api_version}/context/{context_id}"
        
        try:
            response = self.session.get(
                endpoint,
                headers=self._get_headers(),
                timeout=self.config.timeout_seconds,
                verify=self.config.verify_ssl
            )
            
            # Handle response
            response.raise_for_status()
            data = response.json()
            
            # Parse result
            return ContextResponse(
                context_id=data.get("context_id"),
                status=ContextStatus(data.get("status", "unknown")),
                result=data.get("result"),
                error=data.get("error"),
                execution_time_ms=data.get("execution_time_ms"),
                created_at=datetime.fromisoformat(data.get("created_at")) if data.get("created_at") else None,
                completed_at=datetime.fromisoformat(data.get("completed_at")) if data.get("completed_at") else None
            )
            
        except requests.RequestException as e:
            logger.error(f"Failed to get context status: {str(e)}")
            raise


class WebSocketTransport(Transport):
    """WebSocket-based transport for real-time communication."""
    
    def __init__(self, config: ClientConfig, auth_provider: Optional[AuthProvider] = None):
        self.config = config
        self.auth_provider = auth_provider
        self.connected = False
        self.ws = None
        self.pending_contexts = {}
        self.message_queue = asyncio.Queue()
        self.loop = asyncio.new_event_loop()
        
        # Start connection in background
        self._connect()
    
    def _connect(self):
        """Establish WebSocket connection."""
        if not self.config.enable_websocket or not self.config.websocket_url:
            logger.warning("WebSocket transport initialized but not enabled in config")
            return
        
        try:
            # Get auth headers if available
            headers = {}
            if self.auth_provider:
                self.auth_provider.refresh_if_needed()
                headers.update(self.auth_provider.get_auth_headers())
            
            # Initialize WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.config.websocket_url,
                header=headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket connection in a separate thread
            import threading
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {str(e)}")
            self.connected = False
    
    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        logger.info("WebSocket connection established")
        self.connected = True
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            context_id = data.get("context_id")
            
            if context_id and context_id in self.pending_contexts:
                # Update pending context with response
                context_response = ContextResponse(
                    context_id=context_id,
                    status=ContextStatus(data.get("status", "unknown")),
                    result=data.get("result"),
                    error=data.get("error"),
                    execution_time_ms=data.get("execution_time_ms"),
                    created_at=datetime.fromisoformat(data.get("created_at")) if data.get("created_at") else None,
                    completed_at=datetime.fromisoformat(data.get("completed_at")) if data.get("completed_at") else None
                )
                
                # Store in queue for processing
                self.loop.call_soon_threadsafe(
                    self.message_queue.put_nowait, 
                    (context_id, context_response)
                )
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {str(e)}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed."""
        logger.info(f"WebSocket connection closed: {close_msg} (code: {close_status_code})")
        self.connected = False
        
        # Attempt reconnection after delay
        time.sleep(5)
        self._connect()
    
    def execute_context(self, request: ContextRequest) -> ContextResponse:
        """Execute a context via WebSocket."""
        # If not connected or disabled, fall back to HTTP
        if not self.connected or not self.ws:
            logger.warning("WebSocket not connected, falling back to HTTP")
            http_transport = HttpTransport(self.config, self.auth_provider)
            return http_transport.execute_context(request)
        
        request.validate()
        
        # Create message payload
        payload = {
            "action": "execute_context",
            "context_name": request.context_name,
            "service_target": request.service_target,
            "parameters": request.parameters,
            "timeout_seconds": request.timeout_seconds,
            "priority": request.priority
        }
        
        if request.idempotency_key:
            payload["idempotency_key"] = request.idempotency_key
        
        try:
            # Send message
            self.ws.send(json.dumps(payload))
            
            # Wait for response with timeout
            start_time = time.time()
            while time.time() - start_time < request.timeout_seconds:
                try:
                    # Check for response in queue
                    context_id, response = self.loop.run_until_complete(
                        asyncio.wait_for(
                            self.message_queue.get(),
                            timeout=1.0
                        )
                    )
                    
                    # If this is our response, return it
                    if context_id == response.context_id:
                        return response
                        
                except asyncio.TimeoutError:
                    # Keep waiting
                    pass
            
            # Timeout reached
            raise TimeoutError(f"Context execution timed out after {request.timeout_seconds} seconds")
            
        except Exception as e:
            logger.error(f"WebSocket context execution failed: {str(e)}")
            # Fall back to HTTP as a last resort
            http_transport = HttpTransport(self.config, self.auth_provider)
            return http_transport.execute_context(request)
    
    def get_context_status(self, context_id: str) -> ContextResponse:
        """Get context status via WebSocket."""
        # If not connected, fall back to HTTP
        if not self.connected or not self.ws:
            logger.warning("WebSocket not connected, falling back to HTTP")
            http_transport = HttpTransport(self.config, self.auth_provider)
            return http_transport.get_context_status(context_id)
        
        # Create message payload
        payload = {
            "action": "get_context_status",
            "context_id": context_id
        }
        
        try:
            # Send message
            self.ws.send(json.dumps(payload))
            
            # Wait for response with timeout
            start_time = time.time()
            while time.time() - start_time < self.config.timeout_seconds:
                try:
                    # Check for response in queue
                    recv_context_id, response = self.loop.run_until_complete(
                        asyncio.wait_for(
                            self.message_queue.get(),
                            timeout=1.0
                        )
                    )
                    
                    # If this is our response, return it
                    if recv_context_id == context_id:
                        return response
                        
                except asyncio.TimeoutError:
                    # Keep waiting
                    pass
            
            # Timeout reached
            raise TimeoutError(f"Get context status timed out after {self.config.timeout_seconds} seconds")
            
        except Exception as e:
            logger.error(f"WebSocket get_context_status failed: {str(e)}")
            # Fall back to HTTP
            http_transport = HttpTransport(self.config, self.auth_provider)
            return http_transport.get_context_status(context_id)


# ===== MCP Client Implementation =====

class MCPClient:
    """Main client for interacting with MCP servers."""
    
    def __init__(
        self, 
        server_url: str,
        auth_provider: Optional[AuthProvider] = None,
        use_websocket: bool = False,
        **config_kwargs
    ):
        # Create configuration
        self.config = ClientConfig(
            server_url=server_url,
            enable_websocket=use_websocket,
            **config_kwargs
        )
        
        # Set authentication provider
        self.auth_provider = auth_provider
        
        # Initialize appropriate transport
        if use_websocket and self.config.websocket_url:
            self.transport = WebSocketTransport(self.config, self.auth_provider)
            logger.info(f"Initialized WebSocket transport to {self.config.websocket_url}")
        else:
            self.transport = HttpTransport(self.config, self.auth_provider)
            logger.info(f"Initialized HTTP transport to {self.config.server_url}")
    
    def execute_context(self, request: ContextRequest) -> ContextResponse:
        """Execute a context on the MCP server."""
        return self.transport.execute_context(request)
    
    def get_context_status(self, context_id: str) -> ContextResponse:
        """Get the status of a context execution."""
        return self.transport.get_context_status(context_id)
    
    def execute_and_wait(self, request: ContextRequest, poll_interval: float = 0.5) -> ContextResponse:
        """Execute a context and wait for completion."""
        # Execute initial request
        response = self.execute_context(request)
        
        # If already completed, return immediately
        if response.status in [ContextStatus.COMPLETED, ContextStatus.FAILED, ContextStatus.CANCELED]:
            return response
        
        # Poll until completion or timeout
        context_id = response.context_id
        elapsed_time = 0
        while elapsed_time < request.timeout_seconds:
            time.sleep(poll_interval)
            elapsed_time += poll_interval
            
            # Get latest status
            response = self.get_context_status(context_id)
            
            # Return if completed
            if response.status in [ContextStatus.COMPLETED, ContextStatus.FAILED, ContextStatus.CANCELED]:
                return response
            
            # Increase poll interval with backoff (max 2 seconds)
            poll_interval = min(poll_interval * 1.5, 2.0)
        
        # Timeout reached
        raise TimeoutError(f"Context execution timed out after {request.timeout_seconds} seconds")


# ===== Helper Functions =====

def create_client_from_env() -> MCPClient:
    """Create a client using environment variables for configuration."""
    server_url = os.environ.get("MCP_SERVER_URL")
    if not server_url:
        raise ValueError("MCP_SERVER_URL environment variable must be set")
    
    # Create auth provider based on available credentials
    auth_provider = None
    if os.environ.get("MCP_API_KEY"):
        auth_provider = ApiKeyAuthProvider(
            api_key=os.environ.get("MCP_API_KEY"),
            header_name=os.environ.get("MCP_API_KEY_HEADER", "X-API-Key")
        )
    elif os.environ.get("MCP_CLIENT_ID") and os.environ.get("MCP_CLIENT_SECRET"):
        auth_provider = JwtAuthProvider(
            token_url=os.environ.get("MCP_TOKEN_URL", f"{server_url}/api/v1/token"),
            client_id=os.environ.get("MCP_CLIENT_ID"),
            client_secret=os.environ.get("MCP_CLIENT_SECRET")
        )
    
    # Create client with optional websocket support
    use_websocket = os.environ.get("MCP_USE_WEBSOCKET", "false").lower() == "true"
    
    return MCPClient(
        server_url=server_url,
        auth_provider=auth_provider,
        use_websocket=use_websocket,
        api_version=os.environ.get("MCP_API_VERSION", "v1"),
        timeout_seconds=int(os.environ.get("MCP_TIMEOUT_SECONDS", "30")),
        max_retries=int(os.environ.get("MCP_MAX_RETRIES", "3")),
        verify_ssl=os.environ.get("MCP_VERIFY_SSL", "true").lower() == "true"
    )


# ===== Usage Examples =====

def example_usage():
    """Demonstrate example usage of the client."""
    # Basic client setup
    client = MCPClient(
        server_url="https://mcp.example.com",
        auth_provider=ApiKeyAuthProvider(api_key="your-api-key")
    )
    
    # Create a context request
    request = ContextRequest(
        context_name="quickbooks.getInvoices",
        service_target="accounting",
        parameters={"customer_id": "12345", "date_range": {"start": "2023-01-01", "end": "2023-12-31"}},
        timeout_seconds=60
    )
    
    try:
        # Execute and wait for result
        response = client.execute_and_wait(request)
        
        # Handle response
        if response.status == ContextStatus.COMPLETED:
            print(f"Context completed successfully: {response.result}")
        else:
            print(f"Context failed: {response.error}")
            
    except Exception as e:
        print(f"Error executing context: {str(e)}")


if __name__ == "__main__":
    # This is just a demonstration - not meant to be run directly
    print("This module demonstrates the MCP client SDK architecture.")
    print("Import and use the classes in your own code instead of running this file directly.") 