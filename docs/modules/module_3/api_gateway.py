#!/usr/bin/env python3
"""
MCP API Gateway Implementation
=============================

This script demonstrates the implementation of a secure API gateway for MCP servers,
including authentication, rate limiting, request validation, and routing.

The API Gateway acts as the entry point for all client requests to the MCP server,
providing consistent security, validation, and monitoring across all endpoints.

Run with:
    python api_gateway.py

Dependencies:
    - fastapi
    - uvicorn
    - redis (optional, for distributed rate limiting)
    - pydantic
    - python-jose (for JWT)
"""

import os
import sys
import time
import json
import uuid
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable

import uvicorn
from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Optional: For production, use Redis for distributed rate limiting
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis not available. Using in-memory rate limiting.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("mcp_api_gateway")

# ===== Models =====

class ContextRequest(BaseModel):
    """Model representing a client context request."""
    context_name: str
    service_target: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 30
    
    @validator('context_name')
    def context_name_must_be_valid(cls, v):
        """Validate context name format."""
        if not v or not isinstance(v, str) or len(v) < 2:
            raise ValueError('context_name must be a valid string')
        return v
    
    @validator('service_target')
    def service_target_must_be_valid(cls, v):
        """Validate service target format."""
        if not v or not isinstance(v, str) or len(v) < 2:
            raise ValueError('service_target must be a valid string')
        return v
    
    @validator('timeout_seconds')
    def timeout_must_be_reasonable(cls, v):
        """Validate timeout is reasonable."""
        if v < 1 or v > 300:  # 5 minute max timeout
            raise ValueError('timeout_seconds must be between 1 and 300')
        return v

class ContextResponse(BaseModel):
    """Model representing a response to a context request."""
    context_id: str
    status: str
    result: Any
    error: Optional[str] = None
    execution_time_ms: int

class ErrorResponse(BaseModel):
    """Model representing an error response."""
    error: str
    detail: Optional[str] = None
    status_code: int

class UserProfile(BaseModel):
    """Model representing a user profile with permissions."""
    user_id: str
    username: str
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    rate_limit: int = 100  # requests per minute

# ===== API Gateway Implementation =====

class MCPApiGateway:
    """
    API Gateway implementation for MCP server.
    
    This class provides:
    1. Authentication and authorization
    2. Rate limiting
    3. Request validation
    4. Request routing
    5. Response formatting
    6. Logging and monitoring
    """
    
    def __init__(self, mcp_server_url: str, redis_url: Optional[str] = None):
        """Initialize the API Gateway."""
        self.app = FastAPI(title="MCP API Gateway", version="1.0.0")
        self.mcp_server_url = mcp_server_url
        self.setup_middleware()
        self.setup_routes()
        
        # Authentication components
        self.api_key_header = APIKeyHeader(name="X-API-Key")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        
        # In-memory user store (for demo, use a database in production)
        self.users = {
            "user1": UserProfile(
                user_id="user1",
                username="demo_user",
                roles=["basic"],
                permissions=["read:*", "execute:basic_commands"],
                rate_limit=100
            ),
            "admin": UserProfile(
                user_id="admin",
                username="admin_user",
                roles=["admin"],
                permissions=["*"],
                rate_limit=500
            )
        }
        
        # API keys (for demo, use a secure database in production)
        self.api_keys = {
            "api-key-12345": "user1",
            "api-key-admin": "admin"
        }
        
        # Rate limiting
        self.rate_limit_store = {}  # In-memory store
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis = redis.Redis.from_url(redis_url)
                # Test connection
                self.redis.ping()
                self.use_redis = True
                logger.info("Using Redis for rate limiting")
            except:
                self.use_redis = False
                logger.warning("Redis connection failed. Using in-memory rate limiting.")
        else:
            self.use_redis = False
    
    def setup_middleware(self):
        """Set up middleware for the API Gateway."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify allowed origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request ID middleware
        @self.app.middleware("http")
        async def add_request_id(request: Request, call_next):
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        
        # Logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            # Get client IP (handle proxies)
            client_ip = request.client.host
            x_forwarded_for = request.headers.get("X-Forwarded-For")
            if x_forwarded_for:
                client_ip = x_forwarded_for.split(",")[0]
            
            # Log request
            logger.info(f"Request {request.state.request_id}: {request.method} {request.url.path} from {client_ip}")
            
            response = await call_next(request)
            
            # Log response time
            process_time = time.time() - start_time
            logger.info(f"Request {request.state.request_id} completed in {process_time:.4f}s with status {response.status_code}")
            
            return response
    
    def setup_routes(self):
        """Set up API routes for the gateway."""
        @self.app.get("/")
        async def root():
            return {"message": "MCP API Gateway", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health_check():
            # In production, check MCP server health
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.post("/token")
        async def get_token(username: str, password: str):
            # Simplified token generation (use proper auth in production)
            if username in self.users:
                # In production, verify password
                user = self.users[username]
                token = f"demo-token-{user.user_id}-{int(time.time())}"
                return {"access_token": token, "token_type": "bearer"}
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        @self.app.post("/context")
        async def execute_context(
            context: ContextRequest, 
            request: Request,
            user: UserProfile = Depends(self.get_current_user)
        ):
            # Apply rate limiting
            await self.check_rate_limit(user)
            
            # Validate permissions for this context
            self.check_permissions(user, context)
            
            # Generate context ID
            context_id = str(uuid.uuid4())
            
            # Log context
            logger.info(f"Processing context {context_id}: {context.context_name} for user {user.username}")
            
            # In a real implementation, forward to MCP server
            # For demo, simulate processing
            await asyncio.sleep(0.5)
            
            result = {
                "context_id": context_id,
                "status": "completed",
                "result": f"Simulated execution of {context.context_name} on {context.service_target}",
                "execution_time_ms": 500,
            }
            
            # Log completion
            logger.info(f"Context {context_id} completed successfully")
            
            return result
        
        @self.app.get("/contexts/{context_id}")
        async def get_context_status(
            context_id: str,
            user: UserProfile = Depends(self.get_current_user)
        ):
            # In a real implementation, check with MCP server
            # For demo, simulate response
            return {
                "context_id": context_id,
                "status": "completed",
                "result": "Context execution completed",
                "execution_time_ms": 500,
            }
    
    async def get_current_user_by_token(self, token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))):
        """Get the current user from an OAuth token."""
        # In production, validate JWT token properly
        # For demo, just parse the demo token format
        try:
            if token.startswith("demo-token-"):
                parts = token.split("-")
                user_id = parts[2]
                if user_id in self.users:
                    return self.users[user_id]
        except:
            pass
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    async def get_current_user_by_api_key(self, api_key: str = Depends(APIKeyHeader(name="X-API-Key"))):
        """Get the current user from an API key."""
        if api_key in self.api_keys:
            user_id = self.api_keys[api_key]
            if user_id in self.users:
                return self.users[user_id]
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    
    async def get_current_user(self, request: Request):
        """Get the current user from either token or API key."""
        # Try API Key first
        auth_header = request.headers.get("X-API-Key")
        if auth_header:
            if auth_header in self.api_keys:
                user_id = self.api_keys[auth_header]
                if user_id in self.users:
                    return self.users[user_id]
        
        # Then try Bearer token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            return await self.get_current_user_by_token(token)
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    
    def check_permissions(self, user: UserProfile, context: ContextRequest):
        """Check if the user has permissions to execute the context."""
        # Admin can do anything
        if "*" in user.permissions:
            return True
        
        # Check for specific context permissions
        specific_perm = f"execute:{context.context_name}"
        service_perm = f"service:{context.service_target}"
        
        if specific_perm in user.permissions or service_perm in user.permissions:
            return True
        
        # Check for wildcard permissions
        for perm in user.permissions:
            if perm.endswith(":*") and context.context_name.startswith(perm[:-2]):
                return True
        
        # Permission denied
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied for context: {context.context_name}"
        )
    
    async def check_rate_limit(self, user: UserProfile):
        """Check and apply rate limiting for the user."""
        user_id = user.user_id
        rate_limit = user.rate_limit
        
        if self.use_redis:
            # Redis-based rate limiting
            current = int(time.time())
            key = f"ratelimit:{user_id}:{current // 60}"
            
            # Increment counter
            count = self.redis.incr(key)
            
            # Set expiry if new key
            if count == 1:
                self.redis.expire(key, 120)  # 2 minutes expiry
            
            if count > rate_limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
        else:
            # In-memory rate limiting
            current = int(time.time())
            minute_key = current // 60
            
            if user_id not in self.rate_limit_store:
                self.rate_limit_store[user_id] = {}
            
            # Clean up old entries
            self.rate_limit_store[user_id] = {
                k: v for k, v in self.rate_limit_store[user_id].items()
                if k >= minute_key - 1
            }
            
            # Check current minute
            if minute_key not in self.rate_limit_store[user_id]:
                self.rate_limit_store[user_id][minute_key] = 0
            
            # Increment counter
            self.rate_limit_store[user_id][minute_key] += 1
            
            # Check limit
            if self.rate_limit_store[user_id][minute_key] > rate_limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API Gateway server."""
        uvicorn.run(self.app, host=host, port=port)

# ===== Example Usage =====

def main():
    """Run the API Gateway as a standalone server."""
    api_gateway = MCPApiGateway(
        mcp_server_url="http://localhost:8001",
        redis_url=os.environ.get("REDIS_URL")
    )
    
    logger.info("Starting API Gateway")
    api_gateway.run(port=8000)

if __name__ == "__main__":
    main() 