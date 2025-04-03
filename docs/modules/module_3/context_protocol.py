#!/usr/bin/env python3
"""
MCP Context Protocol Specification
================================

This module defines the standard context protocol for Multi-Context Protocol (MCP) servers.
It includes:
- Context structure and validation
- Context serialization and deserialization
- Context routing information
- Response formatting
- Error handling

This standardization ensures consistent context handling across different MCP servers
and clients.
"""

import json
import uuid
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator

# ===== Context Protocol Schema =====

class ContextStatus(str, Enum):
    """Status of a context during its lifecycle."""
    CREATED = "created"
    VALIDATED = "validated"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMEOUT = "timeout"

class ContextPriority(str, Enum):
    """Priority level for context execution."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class ContextSource(str, Enum):
    """Source of the context."""
    USER = "user"
    SYSTEM = "system"
    SERVICE = "service"
    SCHEDULED = "scheduled"

class ErrorCode(str, Enum):
    """Standardized error codes for context failures."""
    VALIDATION_ERROR = "validation_error"
    PERMISSION_DENIED = "permission_denied"
    SERVICE_UNAVAILABLE = "service_unavailable"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    INVALID_FORMAT = "invalid_format"
    INTERNAL_ERROR = "internal_error"

class ContextParameter(BaseModel):
    """Definition of a context parameter."""
    name: str
    type: str
    required: bool = True
    default: Optional[Any] = None
    description: str = ""
    validation: Optional[Dict[str, Any]] = None

class ContextDefinition(BaseModel):
    """Definition of a context supported by a service."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    parameters: List[ContextParameter] = Field(default_factory=list)
    required_permissions: List[str] = Field(default_factory=list)
    service_id: str
    timeout_seconds: int = 60
    idempotent: bool = False
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "parameters": [p.dict() for p in self.parameters],
            "required_permissions": self.required_permissions,
            "service_id": self.service_id,
            "timeout_seconds": self.timeout_seconds,
            "idempotent": self.idempotent
        }

class Context(BaseModel):
    """Base model for MCP contexts."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str = "1.0.0"
    service_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    status: ContextStatus = ContextStatus.CREATED
    priority: ContextPriority = ContextPriority.NORMAL
    source: ContextSource = ContextSource.USER
    user_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    timeout_seconds: int = 60
    
    @validator('name')
    def name_must_be_valid(cls, v):
        """Validate context name."""
        if not v or not isinstance(v, str) or len(v) < 2:
            raise ValueError('context name must be valid')
        return v
    
    @validator('service_id')
    def service_id_must_be_valid(cls, v):
        """Validate service ID."""
        if not v or not isinstance(v, str) or len(v) < 2:
            raise ValueError('service_id must be valid')
        return v
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "service_id": self.service_id,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "priority": self.priority,
            "source": self.source,
            "user_id": self.user_id,
            "idempotency_key": self.idempotency_key,
            "timeout_seconds": self.timeout_seconds
        }
    
    def to_json(self):
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str):
        """Create Context from JSON string."""
        data = json.loads(json_str)
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

class ContextResult(BaseModel):
    """Result of a context execution."""
    context_id: str
    status: ContextStatus
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "context_id": self.context_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat()
        }
    
    def to_json(self):
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str):
        """Create ContextResult from JSON string."""
        data = json.loads(json_str)
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)
    
    @classmethod
    def create_error(cls, context_id: str, error_code: ErrorCode, message: str):
        """Create an error result."""
        return cls(
            context_id=context_id,
            status=ContextStatus.FAILED,
            error={
                "code": error_code,
                "message": message
            }
        )

# ===== Context Validation Functions =====

def validate_context_parameters(context: Context, definition: ContextDefinition) -> Optional[str]:
    """
    Validate context parameters against context definition.
    
    Returns None if validation passes, or error message if it fails.
    """
    # Check that all required parameters are present
    for param in definition.parameters:
        if param.required and param.name not in context.parameters:
            return f"Missing required parameter: {param.name}"
    
    # Check parameter types and validation rules
    for param_name, param_value in context.parameters.items():
        # Find parameter definition
        param_def = next((p for p in definition.parameters if p.name == param_name), None)
        
        # If parameter is not in definition, it's an unknown parameter
        if not param_def:
            return f"Unknown parameter: {param_name}"
        
        # Type checking (simplified)
        if param_def.type == "string" and not isinstance(param_value, str):
            return f"Parameter {param_name} must be a string"
        elif param_def.type == "integer" and not isinstance(param_value, int):
            return f"Parameter {param_name} must be an integer"
        elif param_def.type == "number" and not isinstance(param_value, (int, float)):
            return f"Parameter {param_name} must be a number"
        elif param_def.type == "boolean" and not isinstance(param_value, bool):
            return f"Parameter {param_name} must be a boolean"
        elif param_def.type == "array" and not isinstance(param_value, list):
            return f"Parameter {param_name} must be an array"
        elif param_def.type == "object" and not isinstance(param_value, dict):
            return f"Parameter {param_name} must be an object"
        
        # Apply custom validation if defined
        if param_def.validation:
            if param_def.type == "string" and param_def.validation.get("min_length"):
                if len(param_value) < param_def.validation["min_length"]:
                    return f"Parameter {param_name} must have minimum length of {param_def.validation['min_length']}"
            
            if param_def.type == "string" and param_def.validation.get("max_length"):
                if len(param_value) > param_def.validation["max_length"]:
                    return f"Parameter {param_name} must have maximum length of {param_def.validation['max_length']}"
            
            if param_def.type in ("integer", "number") and "min_value" in param_def.validation:
                if param_value < param_def.validation["min_value"]:
                    return f"Parameter {param_name} must be at least {param_def.validation['min_value']}"
            
            if param_def.type in ("integer", "number") and "max_value" in param_def.validation:
                if param_value > param_def.validation["max_value"]:
                    return f"Parameter {param_name} must be at most {param_def.validation['max_value']}"
            
            if param_def.type == "string" and "pattern" in param_def.validation:
                import re
                if not re.match(param_def.validation["pattern"], param_value):
                    return f"Parameter {param_name} does not match required pattern"
            
            if "enum" in param_def.validation and param_value not in param_def.validation["enum"]:
                return f"Parameter {param_name} must be one of: {', '.join(str(v) for v in param_def.validation['enum'])}"
    
    return None  # Validation passed

# ===== Example Context Definitions =====

def get_example_context_definitions():
    """Return example context definitions for common services."""
    return [
        ContextDefinition(
            name="generate_text",
            description="Generate text using an LLM",
            service_id="llm-service",
            parameters=[
                ContextParameter(
                    name="prompt",
                    type="string",
                    required=True,
                    description="The text prompt to generate from",
                    validation={"min_length": 1, "max_length": 4000}
                ),
                ContextParameter(
                    name="max_tokens",
                    type="integer",
                    required=False,
                    default=100,
                    description="Maximum number of tokens to generate",
                    validation={"min_value": 1, "max_value": 2000}
                ),
                ContextParameter(
                    name="temperature",
                    type="number",
                    required=False,
                    default=0.7,
                    description="Sampling temperature",
                    validation={"min_value": 0.0, "max_value": 2.0}
                )
            ],
            required_permissions=["llm:generate"],
            timeout_seconds=120
        ),
        ContextDefinition(
            name="process_image",
            description="Process an image using AI",
            service_id="vision-service",
            parameters=[
                ContextParameter(
                    name="image_url",
                    type="string",
                    required=True,
                    description="URL of the image to process",
                    validation={"min_length": 5}
                ),
                ContextParameter(
                    name="operations",
                    type="array",
                    required=True,
                    description="List of operations to perform",
                    validation={"enum": ["resize", "crop", "analyze", "detect_objects", "caption"]}
                ),
                ContextParameter(
                    name="options",
                    type="object",
                    required=False,
                    description="Additional options for processing"
                )
            ],
            required_permissions=["vision:process"],
            timeout_seconds=180
        ),
        ContextDefinition(
            name="query_database",
            description="Query a vector database",
            service_id="database-service",
            parameters=[
                ContextParameter(
                    name="query",
                    type="string",
                    required=True,
                    description="Query string or vector"
                ),
                ContextParameter(
                    name="collection",
                    type="string",
                    required=True,
                    description="Collection to query"
                ),
                ContextParameter(
                    name="limit",
                    type="integer",
                    required=False,
                    default=10,
                    description="Maximum number of results to return",
                    validation={"min_value": 1, "max_value": 100}
                )
            ],
            required_permissions=["db:query"],
            timeout_seconds=30,
            idempotent=True
        )
    ]

# ===== Example Usage =====

def create_example_context():
    """Create an example context for demonstration."""
    return Context(
        name="generate_text",
        service_id="llm-service",
        parameters={
            "prompt": "Explain the MCP context protocol",
            "max_tokens": 150,
            "temperature": 0.7
        },
        user_id="user-123"
    )

def example_context_lifecycle():
    """Demonstrate a context lifecycle."""
    # Create a context
    context = create_example_context()
    print(f"Context created: {context.id}")
    
    # Serialize for transmission
    context_json = context.to_json()
    print(f"Serialized context: {context_json}")
    
    # Deserialize (e.g., on the MCP server)
    received_context = Context.from_json(context_json)
    print(f"Deserialized context: {received_context.id}")
    
    # Validate context
    context_defs = get_example_context_definitions()
    llm_context_def = context_defs[0]  # The generate_text definition
    
    validation_error = validate_context_parameters(received_context, llm_context_def)
    if validation_error:
        print(f"Validation error: {validation_error}")
        result = ContextResult.create_error(
            context_id=received_context.id,
            error_code=ErrorCode.VALIDATION_ERROR,
            message=validation_error
        )
    else:
        print("Context validation passed")
        received_context.status = ContextStatus.VALIDATED
        
        # Simulate execution
        start_time = time.time()
        print("Executing context...")
        time.sleep(1)  # Simulate processing time
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Create successful result
        result = ContextResult(
            context_id=received_context.id,
            status=ContextStatus.COMPLETED,
            result="The MCP context protocol standardizes how contexts are sent, validated, and processed...",
            execution_time_ms=execution_time_ms
        )
    
    # Serialize result for transmission
    result_json = result.to_json()
    print(f"Result: {result_json}")
    
    return result

# ===== Main Function =====

if __name__ == "__main__":
    print("MCP Context Protocol Example")
    print("============================")
    
    # Show example context definitions
    print("\nExample Context Definitions:")
    for ctx_def in get_example_context_definitions():
        print(f"- {ctx_def.name} (service: {ctx_def.service_id})")
    
    # Demonstrate context lifecycle
    print("\nContext Lifecycle Demonstration:")
    result = example_context_lifecycle() 