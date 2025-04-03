#!/usr/bin/env python3
"""
MCP Authentication & Authorization Patterns
=========================================

This module demonstrates various authentication and authorization patterns 
for MCP servers, including:

1. API Key Authentication
2. JWT-based Authentication
3. OAuth 2.0 Integration
4. Role-Based Access Control (RBAC)
5. Service-to-Service Authentication
6. Multi-factor Authentication

Run with:
    python auth_patterns.py

Dependencies:
    - fastapi
    - python-jose[cryptography]
    - passlib[bcrypt]
    - pydantic
"""

import os
import sys
import json
import time
import uuid
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Security, Request, Response
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
    APIKeyHeader,
    APIKeyCookie,
    APIKeyQuery,
    SecurityScopes,
)
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("mcp_auth_patterns")

# ===== Configuration =====

# Security settings (in production, store securely)
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# API key settings
API_KEY_NAME = "X-API-Key"

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "read:commands": "Read information about commands",
        "execute:commands": "Execute commands",
        "admin": "Admin access",
    },
)

# API Key security schemes
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)
api_key_cookie = APIKeyCookie(name=API_KEY_NAME, auto_error=False)

# ===== Models =====

class Token(BaseModel):
    """Model for access token response."""
    access_token: str
    token_type: str
    expires_at: datetime
    scope: str = ""

class TokenData(BaseModel):
    """Model for decoded token data."""
    username: Optional[str] = None
    scopes: List[str] = []
    expires_at: Optional[datetime] = None

class UserInDB(BaseModel):
    """User model as stored in the database."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    hashed_password: str
    scopes: List[str] = []
    roles: List[str] = []

class User(BaseModel):
    """User model without sensitive information."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[str] = []
    scopes: List[str] = []

class Role(BaseModel):
    """Role definition with permissions."""
    name: str
    description: str
    permissions: List[str] = []

class ApiKey(BaseModel):
    """API key model."""
    key: str
    user_id: str
    created_at: datetime
    last_used: Optional[datetime] = None
    description: str = ""
    expires_at: Optional[datetime] = None
    scopes: List[str] = []

class ServiceCredential(BaseModel):
    """Service-to-service authentication credential."""
    service_id: str
    service_name: str
    api_key: str
    allowed_endpoints: List[str] = []

class MFAMethod(str, Enum):
    """Multi-factor authentication methods."""
    TOTP = "totp"  # Time-based One-Time Password
    SMS = "sms"    # SMS verification
    EMAIL = "email"  # Email verification

class UserMFA(BaseModel):
    """Multi-factor authentication settings for a user."""
    user_id: str
    mfa_enabled: bool = False
    mfa_method: Optional[MFAMethod] = None
    mfa_verified: bool = False
    mfa_secret: Optional[str] = None

# ===== Mock Database =====

# In a real application, these would be in a database
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": pwd_context.hash("secret"),
        "disabled": False,
        "roles": ["user"],
        "scopes": ["read:commands", "execute:commands"],
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderland",
        "email": "alice@example.com",
        "hashed_password": pwd_context.hash("alice123"),
        "disabled": False,
        "roles": ["user"],
        "scopes": ["read:commands"],
    },
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("admin123"),
        "disabled": False,
        "roles": ["admin"],
        "scopes": ["read:commands", "execute:commands", "admin"],
    },
}

# API Keys database
api_keys_db = {
    "api-key-12345": {
        "key": "api-key-12345",
        "user_id": "johndoe",
        "created_at": datetime.now() - timedelta(days=30),
        "last_used": datetime.now() - timedelta(days=1),
        "description": "John's API key",
        "expires_at": datetime.now() + timedelta(days=60),
        "scopes": ["read:commands", "execute:commands"],
    },
    "api-key-admin": {
        "key": "api-key-admin",
        "user_id": "admin",
        "created_at": datetime.now() - timedelta(days=10),
        "last_used": datetime.now(),
        "description": "Admin API key",
        "expires_at": None,  # Never expires
        "scopes": ["read:commands", "execute:commands", "admin"],
    }
}

# Roles database with permissions
roles_db = {
    "user": {
        "name": "user",
        "description": "Regular user with basic access",
        "permissions": [
            "commands:list",
            "commands:execute:basic",
            "services:list",
        ]
    },
    "power_user": {
        "name": "power_user",
        "description": "Power user with extended access",
        "permissions": [
            "commands:list",
            "commands:execute:*",
            "services:list",
        ]
    },
    "admin": {
        "name": "admin",
        "description": "Administrator with full access",
        "permissions": ["*"]  # Wildcard for all permissions
    }
}

# Service credentials for service-to-service auth
service_credentials_db = {
    "llm-service-1": {
        "service_id": "llm-service-1",
        "service_name": "GPT-4 Turbo Service",
        "api_key": "service-key-12345",
        "allowed_endpoints": [
            "/mcp/commands", 
            "/mcp/services/register"
        ]
    },
    "tool-service-1": {
        "service_id": "tool-service-1",
        "service_name": "Image Processing Service",
        "api_key": "service-key-67890",
        "allowed_endpoints": [
            "/mcp/commands", 
            "/mcp/services/register"
        ]
    }
}

# MFA settings for users
user_mfa_db = {
    "johndoe": {
        "user_id": "johndoe",
        "mfa_enabled": True,
        "mfa_method": MFAMethod.TOTP,
        "mfa_verified": True,
        "mfa_secret": "JBSWY3DPEHPK3PXP",  # Demo secret, never hardcode in production
    },
    "admin": {
        "user_id": "admin",
        "mfa_enabled": False,
        "mfa_method": None,
        "mfa_verified": False,
        "mfa_secret": None,
    }
}

# ===== Helper Functions =====

def verify_password(plain_password, hashed_password):
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Generate a password hash."""
    return pwd_context.hash(password)

def get_user(db, username: str):
    """Get a user from the database."""
    if username in db:
        user_data = db[username]
        return UserInDB(**user_data)
    return None

def authenticate_user(fake_db, username: str, password: str):
    """Authenticate a user with username and password."""
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, expire

def get_api_key(key: str):
    """Get API key details from the database."""
    if key in api_keys_db:
        api_key_data = api_keys_db[key]
        return ApiKey(**api_key_data)
    return None

def validate_api_key(key: str):
    """Validate an API key."""
    api_key = get_api_key(key)
    if not api_key:
        return None
    
    # Check if expired
    if api_key.expires_at and api_key.expires_at < datetime.now():
        return None
    
    # Update last used time (in a real app, update the database)
    api_key.last_used = datetime.now()
    
    # Get user
    user = get_user(fake_users_db, api_key.user_id)
    if not user or user.disabled:
        return None
    
    return user

def generate_api_key():
    """Generate a new API key."""
    return f"api-key-{secrets.token_urlsafe(16)}"

def generate_service_key():
    """Generate a new service-to-service API key."""
    return f"service-key-{secrets.token_urlsafe(20)}"

def has_permission(user: User, required_permission: str):
    """Check if a user has the required permission based on roles."""
    if not user or not user.roles:
        return False
    
    # Check each role
    for role_name in user.roles:
        if role_name not in roles_db:
            continue
        
        role = roles_db[role_name]
        
        # Admin role has all permissions
        if "*" in role["permissions"]:
            return True
        
        # Direct permission match
        if required_permission in role["permissions"]:
            return True
        
        # Wildcard permission match
        for permission in role["permissions"]:
            if permission.endswith("*"):
                prefix = permission[:-1]
                if required_permission.startswith(prefix):
                    return True
    
    return False

def user_from_token(token: str):
    """Extract user from a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        
        token_scopes = payload.get("scopes", [])
        token_exp = datetime.fromtimestamp(payload.get("exp"))
        
        if token_exp < datetime.now():
            return None
            
        token_data = TokenData(
            username=username, 
            scopes=token_scopes,
            expires_at=token_exp
        )
    except JWTError:
        return None
    
    user = get_user(fake_users_db, token_data.username)
    if user is None:
        return None
    
    if user.disabled:
        return None
        
    return user

# ===== FastAPI App & Dependencies =====

app = FastAPI(title="MCP Authentication Patterns")

async def get_current_user_from_token(
    security_scopes: SecurityScopes, token: str = Depends(oauth2_scheme)
):
    """Get the current user from a JWT token with scope checking."""
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
            
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(username=username, scopes=token_scopes)
    except JWTError:
        raise credentials_exception
        
    user = get_user(fake_users_db, token_data.username)
    if user is None:
        raise credentials_exception
        
    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    # Check scopes
    for scope in security_scopes.scopes:
        if scope not in token_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required: {scope}",
                headers={"WWW-Authenticate": authenticate_value},
            )
    
    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=user.disabled,
        roles=user.roles,
        scopes=user.scopes
    )

async def get_current_user_from_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query),
    api_key_cookie: str = Security(api_key_cookie),
):
    """Get the current user from an API key (header, query or cookie)."""
    # Try to get the API key from header, query, or cookie
    api_key = api_key_header or api_key_query or api_key_cookie
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
        )
    
    # Validate the API key
    user = validate_api_key(api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=user.disabled,
        roles=user.roles,
        scopes=user.scopes
    )

async def get_current_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key),
):
    """Get the current user from either token or API key."""
    if token_user:
        return token_user
    if api_key_user:
        return api_key_user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Check if the current user is active."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def verify_service_auth(
    service_id: str, 
    api_key: str = Security(api_key_header), 
    request: Request = None
):
    """Verify service-to-service authentication."""
    if not api_key or service_id not in service_credentials_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Service authentication failed",
        )
    
    service_cred = service_credentials_db[service_id]
    if service_cred["api_key"] != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service API key",
        )
    
    # Check endpoint restriction if request is provided
    if request and request.url.path not in service_cred["allowed_endpoints"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Service {service_id} is not allowed to access this endpoint",
        )
    
    return service_cred

# ===== Routes =====

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Endpoint to get a JWT access token."""
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if MFA is enabled for this user
    if user.username in user_mfa_db and user_mfa_db[user.username]["mfa_enabled"]:
        # In a real app, this would trigger MFA verification
        # For this demo, we'll assume MFA is already verified
        pass
    
    # Filter requested scopes to only those the user has
    scopes = [scope for scope in form_data.scopes if scope in user.scopes]
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token, expires_at = create_access_token(
        data={"sub": user.username, "scopes": scopes},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": expires_at,
        "scope": " ".join(scopes)
    }

@app.post("/api-keys", response_model=ApiKey)
async def create_api_key(
    description: str, 
    expires_days: Optional[int] = 365,
    current_user: User = Security(
        get_current_active_user, 
        scopes=["admin"]
    )
):
    """Create a new API key (admin only)."""
    api_key = generate_api_key()
    
    # In a real app, save to database
    expires_at = datetime.now() + timedelta(days=expires_days) if expires_days else None
    
    new_key = ApiKey(
        key=api_key,
        user_id=current_user.username,
        created_at=datetime.now(),
        description=description,
        expires_at=expires_at,
        scopes=current_user.scopes
    )
    
    # In this demo, we just return the key
    # In a real app, save to database
    return new_key

@app.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get the current user's information."""
    return current_user

@app.get("/status")
async def read_system_status(
    current_user: User = Security(
        get_current_active_user, 
        scopes=["read:commands"]
    )
):
    """Get system status (requires read:commands scope)."""
    return {
        "status": "operational",
        "user": current_user.username,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/commands/execute")
async def execute_command(
    command: str,
    params: Dict[str, Any],
    current_user: User = Security(
        get_current_active_user, 
        scopes=["execute:commands"]
    )
):
    """Execute a command (requires execute:commands scope)."""
    # Check RBAC permission
    required_permission = f"commands:execute:{command}"
    if not has_permission(current_user, required_permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User does not have permission: {required_permission}",
        )
    
    # In a real app, this would execute the command
    return {
        "command": command,
        "params": params,
        "status": "executed",
        "result": f"Simulated execution of {command} by {current_user.username}"
    }

@app.post("/service/register")
async def register_service(
    service_name: str,
    service_type: str,
    service_url: str,
    admin_user: User = Security(
        get_current_active_user, 
        scopes=["admin"]
    )
):
    """Register a new service (admin only)."""
    service_id = f"{service_type}-{uuid.uuid4().hex[:8]}"
    service_key = generate_service_key()
    
    # In a real app, save to database
    return {
        "service_id": service_id,
        "service_name": service_name,
        "api_key": service_key,
        "message": f"Service registered successfully. Keep this API key secure!"
    }

@app.post("/service/{service_id}/command")
async def service_execute_command(
    service_id: str,
    command: str,
    params: Dict[str, Any],
    request: Request,
    service_cred: Dict = Depends(verify_service_auth)
):
    """Execute a command from another service (service-to-service auth)."""
    return {
        "service": service_id,
        "command": command,
        "params": params,
        "status": "executed",
        "result": f"Simulated execution of {command} by service {service_id}"
    }

@app.post("/mfa/enable")
async def enable_mfa(
    mfa_method: MFAMethod,
    current_user: User = Depends(get_current_active_user)
):
    """Enable MFA for the current user."""
    # In a real app, generate and store MFA secret
    # For TOTP, generate QR code for the user to scan
    
    secret = "JBSWY3DPEHPK3PXP"  # Demo secret, use proper generation in production
    
    # In a real app, save to database and require verification
    return {
        "message": f"MFA {mfa_method.value} enabled successfully",
        "secret": secret,
        "verification_required": True
    }

@app.post("/mfa/verify")
async def verify_mfa(
    code: str,
    current_user: User = Depends(get_current_active_user)
):
    """Verify MFA code during login or setup."""
    # In a real app, verify the provided code against the stored secret
    # This is just a demo implementation
    
    if code == "123456":  # Demo code, use proper verification in production
        return {"message": "MFA verified successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid MFA code",
        )

# ===== Example Code Snippets =====

def jwt_auth_example():
    """Example code snippet for JWT authentication."""
    return """
    # Client-side JWT authentication example
    
    import requests
    
    # Step 1: Obtain a JWT token
    token_response = requests.post(
        "https://api.example.com/token",
        data={"username": "johndoe", "password": "secret", "scope": "read:commands execute:commands"}
    )
    token_data = token_response.json()
    access_token = token_data["access_token"]
    
    # Step 2: Use the token for API requests
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get("https://api.example.com/commands", headers=headers)
    
    if response.status_code == 401:
        # Token expired, get a new one
        # ...
    """

def api_key_auth_example():
    """Example code snippet for API key authentication."""
    return """
    # Client-side API key authentication example
    
    import requests
    
    # Use API key in header
    api_key = "api-key-12345"
    headers = {"X-API-Key": api_key}
    response = requests.get("https://api.example.com/commands", headers=headers)
    
    # Alternatively, use API key in query string
    response = requests.get(
        "https://api.example.com/commands",
        params={"X-API-Key": api_key}
    )
    """

def mfa_login_example():
    """Example code snippet for MFA login."""
    return """
    # Client-side MFA login example
    
    import requests
    
    # Step 1: Regular login
    token_response = requests.post(
        "https://api.example.com/token",
        data={"username": "johndoe", "password": "secret"}
    )
    
    # Step 2: Check if MFA is required
    if token_response.status_code == 200 and token_response.json().get("mfa_required"):
        # Step 3: Get MFA code from user
        mfa_code = input("Enter your MFA code: ")
        
        # Step 4: Submit MFA verification
        mfa_response = requests.post(
            "https://api.example.com/mfa/verify",
            json={"code": mfa_code},
            headers={"Authorization": f"Bearer {token_response.json()['access_token']}"}
        )
        
        if mfa_response.status_code == 200:
            # MFA successful, get the final token
            final_token = mfa_response.json()["access_token"]
        else:
            print("MFA verification failed")
    else:
        # No MFA required or login failed
        final_token = token_response.json().get("access_token")
    """

def service_auth_example():
    """Example code snippet for service-to-service authentication."""
    return """
    # Service-to-service authentication example
    
    import requests
    
    service_id = "llm-service-1"
    service_key = "service-key-12345"
    
    # Call another MCP service
    response = requests.post(
        "https://api.example.com/service/tool-service-1/command",
        headers={"X-API-Key": service_key},
        json={
            "command": "process_image",
            "params": {"url": "https://example.com/image.jpg"}
        }
    )
    """

# ===== Main Function =====

def main():
    """Run the auth patterns demo server."""
    # Print some example code snippets
    print("\n=== JWT Authentication Example ===")
    print(jwt_auth_example())
    
    print("\n=== API Key Authentication Example ===")
    print(api_key_auth_example())
    
    print("\n=== MFA Login Example ===")
    print(mfa_login_example())
    
    print("\n=== Service-to-Service Authentication Example ===")
    print(service_auth_example())
    
    # Start the demo server
    print("\nStarting auth patterns demo server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 