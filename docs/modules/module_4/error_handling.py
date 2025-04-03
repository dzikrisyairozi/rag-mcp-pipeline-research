#!/usr/bin/env python3
"""
MCP Error Handling & Validation
=============================

This module provides robust error handling and validation tools for
MCP client implementations. It includes exception classes, error codes,
validation helpers, and strategies for handling errors gracefully.

Key components:
- ExceptionHierarchy: Well-defined exception hierarchy
- ValidationHelpers: Tools for validating context parameters
- ErrorHandlingStrategies: Different strategies for handling errors
- ErrorFormatter: Standardized error message formatting
"""

import inspect
import logging
import traceback
import json
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Callable, Type, TypeVar, Generic
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mcp_error_handling')


# ===== Error Codes =====

class ErrorCategory(Enum):
    """Categories of errors in the MCP system."""
    AUTHENTICATION = "auth"
    AUTHORIZATION = "authz"
    VALIDATION = "validation"
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    SERVICE = "service"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class ErrorCode(Enum):
    """Specific error codes in the MCP system."""
    # Authentication errors
    AUTH_INVALID_CREDENTIALS = "auth_invalid_credentials"
    AUTH_EXPIRED_TOKEN = "auth_expired_token"
    AUTH_MISSING_TOKEN = "auth_missing_token"
    
    # Authorization errors
    AUTHZ_INSUFFICIENT_PERMISSIONS = "authz_insufficient_permissions"
    AUTHZ_RESOURCE_FORBIDDEN = "authz_resource_forbidden"
    
    # Validation errors
    VALIDATION_MISSING_PARAMETER = "validation_missing_parameter"
    VALIDATION_INVALID_PARAMETER = "validation_invalid_parameter"
    VALIDATION_INVALID_CONTEXT = "validation_invalid_context"
    
    # Connection errors
    CONNECTION_FAILED = "connection_failed"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_REJECTED = "connection_rejected"
    
    # Timeout errors
    TIMEOUT_CONNECTION = "timeout_connection"
    TIMEOUT_OPERATION = "timeout_operation"
    TIMEOUT_IDLE = "timeout_idle"
    
    # Service errors
    SERVICE_UNAVAILABLE = "service_unavailable"
    SERVICE_RATE_LIMITED = "service_rate_limited"
    SERVICE_DEPENDENCY_FAILED = "service_dependency_failed"
    
    # Internal errors
    INTERNAL_ERROR = "internal_error"
    INTERNAL_DATA_ERROR = "internal_data_error"
    
    # Unknown errors
    UNKNOWN_ERROR = "unknown_error"
    
    @classmethod
    def get_category(cls, code: 'ErrorCode') -> ErrorCategory:
        """Get the category for a specific error code."""
        code_str = code.value
        
        if code_str.startswith("auth_"):
            return ErrorCategory.AUTHENTICATION
        elif code_str.startswith("authz_"):
            return ErrorCategory.AUTHORIZATION
        elif code_str.startswith("validation_"):
            return ErrorCategory.VALIDATION
        elif code_str.startswith("connection_"):
            return ErrorCategory.CONNECTION
        elif code_str.startswith("timeout_"):
            return ErrorCategory.TIMEOUT
        elif code_str.startswith("service_"):
            return ErrorCategory.SERVICE
        elif code_str.startswith("internal_"):
            return ErrorCategory.INTERNAL
        else:
            return ErrorCategory.UNKNOWN


# ===== Exception Hierarchy =====

class MCPError(Exception):
    """Base exception for all MCP errors."""
    
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize MCP error."""
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause
        super().__init__(message)
    
    @property
    def category(self) -> ErrorCategory:
        """Get the category of this error."""
        return ErrorCode.get_category(self.code)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        result = {
            "message": self.message,
            "code": self.code.value,
            "category": self.category.value
        }
        
        if self.details:
            result["details"] = self.details
            
        if self.cause:
            result["cause"] = str(self.cause)
            
        return result
    
    def __str__(self) -> str:
        """Get string representation of error."""
        return f"{self.code.value}: {self.message}"


class AuthenticationError(MCPError):
    """Error related to authentication."""
    
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.AUTH_INVALID_CREDENTIALS,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize authentication error."""
        super().__init__(message, code, details, cause)


class AuthorizationError(MCPError):
    """Error related to authorization."""
    
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.AUTHZ_INSUFFICIENT_PERMISSIONS,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize authorization error."""
        super().__init__(message, code, details, cause)


class ValidationError(MCPError):
    """Error related to validation."""
    
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.VALIDATION_INVALID_PARAMETER,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize validation error."""
        super().__init__(message, code, details, cause)


class ConnectionError(MCPError):
    """Error related to connection."""
    
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.CONNECTION_FAILED,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize connection error."""
        super().__init__(message, code, details, cause)


class TimeoutError(MCPError):
    """Error related to timeouts."""
    
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.TIMEOUT_OPERATION,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize timeout error."""
        super().__init__(message, code, details, cause)


class ServiceError(MCPError):
    """Error related to services."""
    
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.SERVICE_UNAVAILABLE,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize service error."""
        super().__init__(message, code, details, cause)


class InternalError(MCPError):
    """Error related to internal problems."""
    
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize internal error."""
        super().__init__(message, code, details, cause)


# ===== Error Registry =====

class ErrorRegistry:
    """Registry of known errors and how to handle them."""
    
    def __init__(self):
        """Initialize error registry."""
        self.handlers: Dict[ErrorCode, Callable[[MCPError], None]] = {}
        self.default_handler: Optional[Callable[[MCPError], None]] = None
    
    def register_handler(self, code: ErrorCode, handler: Callable[[MCPError], None]):
        """Register a handler for a specific error code."""
        self.handlers[code] = handler
    
    def register_default_handler(self, handler: Callable[[MCPError], None]):
        """Register a default handler for errors."""
        self.default_handler = handler
    
    def handle(self, error: MCPError):
        """Handle an error using the appropriate handler."""
        handler = self.handlers.get(error.code)
        
        if handler:
            handler(error)
        elif self.default_handler:
            self.default_handler(error)
        else:
            # If no handler is registered, log the error
            logger.error(f"Unhandled error: {error}")


# ===== Validation Helpers =====

class ValidationHelper:
    """Helper for validating context parameters."""
    
    @staticmethod
    def validate_required(params: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """Validate that required keys are present in parameters."""
        missing = []
        for key in required_keys:
            if key not in params or params[key] is None:
                missing.append(key)
        return missing
    
    @staticmethod
    def validate_type(value: Any, expected_type: Type, key: str) -> Optional[str]:
        """Validate that a value is of the expected type."""
        if not isinstance(value, expected_type):
            return f"{key} must be of type {expected_type.__name__}, got {type(value).__name__}"
        return None
    
    @staticmethod
    def validate_enum(value: Any, enum_class: Type[Enum], key: str) -> Optional[str]:
        """Validate that a value is a valid enum value."""
        try:
            enum_class(value)
            return None
        except ValueError:
            valid_values = [e.value for e in enum_class]
            return f"{key} must be one of {valid_values}, got {value}"
    
    @staticmethod
    def validate_range(value: Union[int, float], min_value: Optional[Union[int, float]], 
                      max_value: Optional[Union[int, float]], key: str) -> Optional[str]:
        """Validate that a value is within a range."""
        if min_value is not None and value < min_value:
            return f"{key} must be at least {min_value}, got {value}"
        if max_value is not None and value > max_value:
            return f"{key} must be at most {max_value}, got {value}"
        return None
    
    @staticmethod
    def validate_length(value: Union[str, List, Dict], min_length: Optional[int], 
                        max_length: Optional[int], key: str) -> Optional[str]:
        """Validate that a value's length is within a range."""
        length = len(value)
        if min_length is not None and length < min_length:
            return f"{key} must have at least {min_length} items, got {length}"
        if max_length is not None and length > max_length:
            return f"{key} must have at most {max_length} items, got {length}"
        return None
    
    @staticmethod
    def validate_format(value: str, format_name: str, key: str) -> Optional[str]:
        """Validate that a string matches a specific format."""
        import re
        
        formats = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "date": r"^\d{4}-\d{2}-\d{2}$",
            "datetime": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$",
            "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            "phone": r"^\+?[0-9\s\-\(\)]+$"
        }
        
        if format_name not in formats:
            return f"Unknown format: {format_name}"
        
        if not re.match(formats[format_name], value):
            return f"{key} must be a valid {format_name}, got {value}"
        
        return None


# ===== Error Handling Strategies =====

class ErrorStrategy:
    """Base class for error handling strategies."""
    
    def handle(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with error handling strategy."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return self._handle_exception(e, func, args, kwargs)
    
    def _handle_exception(self, exception: Exception, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Handle an exception based on the strategy."""
        raise NotImplementedError("Subclasses must implement this method")


class LogAndRaiseStrategy(ErrorStrategy):
    """Strategy that logs the error and re-raises it."""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """Initialize log and raise strategy."""
        self.logger = logger_instance or logger
    
    def _handle_exception(self, exception: Exception, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Log the error and re-raise it."""
        # Convert to MCP error if it's not already
        if not isinstance(exception, MCPError):
            mcp_error = InternalError(
                message=str(exception),
                code=ErrorCode.INTERNAL_ERROR,
                details={
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                },
                cause=exception
            )
        else:
            mcp_error = exception
        
        # Log the error
        self.logger.error(
            f"Error in {func.__name__}: {mcp_error}",
            exc_info=True
        )
        
        # Re-raise the error
        raise mcp_error


class RetryStrategy(ErrorStrategy):
    """Strategy that retries the function on error."""
    
    def __init__(
        self, 
        max_retries: int = 3, 
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        retry_on: Optional[List[Type[Exception]]] = None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """Initialize retry strategy."""
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.retry_on = retry_on or [ConnectionError, TimeoutError, ServiceError]
        self.logger = logger_instance or logger
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if the exception should trigger a retry."""
        return any(isinstance(exception, exc_type) for exc_type in self.retry_on)
    
    def _handle_exception(self, exception: Exception, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Retry the function if appropriate."""
        import time
        
        # Convert to MCP error if it's not already
        if not isinstance(exception, MCPError):
            mcp_error = InternalError(
                message=str(exception),
                code=ErrorCode.INTERNAL_ERROR,
                details={
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                },
                cause=exception
            )
        else:
            mcp_error = exception
        
        # Check if we should retry
        if not self._should_retry(mcp_error):
            self.logger.error(
                f"Error in {func.__name__}, not retrying: {mcp_error}",
                exc_info=True
            )
            raise mcp_error
        
        # Try to retry
        for attempt in range(1, self.max_retries + 1):
            delay = self.retry_delay * (self.backoff_factor ** (attempt - 1))
            
            self.logger.warning(
                f"Error in {func.__name__}, retrying in {delay:.2f}s (attempt {attempt}/{self.max_retries}): {mcp_error}"
            )
            
            time.sleep(delay)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    # Convert to MCP error if it's not already
                    if not isinstance(e, MCPError):
                        final_error = InternalError(
                            message=f"Failed after {self.max_retries} retries: {str(e)}",
                            code=ErrorCode.INTERNAL_ERROR,
                            details={
                                "function": func.__name__,
                                "args": str(args),
                                "kwargs": str(kwargs),
                                "attempts": self.max_retries
                            },
                            cause=e
                        )
                    else:
                        final_error = e
                    
                    self.logger.error(
                        f"Error in {func.__name__}, retry limit exceeded: {final_error}",
                        exc_info=True
                    )
                    
                    raise final_error


class FallbackStrategy(ErrorStrategy):
    """Strategy that falls back to a default value or function on error."""
    
    def __init__(
        self, 
        fallback_value: Any = None,
        fallback_function: Optional[Callable] = None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """Initialize fallback strategy."""
        self.fallback_value = fallback_value
        self.fallback_function = fallback_function
        self.logger = logger_instance or logger
    
    def _handle_exception(self, exception: Exception, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Return fallback value or result of fallback function."""
        # Convert to MCP error if it's not already
        if not isinstance(exception, MCPError):
            mcp_error = InternalError(
                message=str(exception),
                code=ErrorCode.INTERNAL_ERROR,
                details={
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                },
                cause=exception
            )
        else:
            mcp_error = exception
        
        # Log the error
        self.logger.warning(
            f"Error in {func.__name__}, using fallback: {mcp_error}",
            exc_info=True
        )
        
        # Return fallback value or result of fallback function
        if self.fallback_function is not None:
            return self.fallback_function(*args, **kwargs)
        else:
            return self.fallback_value


# ===== Error Formatter =====

class ErrorFormatter:
    """Formats error messages for different outputs."""
    
    @staticmethod
    def to_json(error: MCPError) -> str:
        """Format error as JSON string."""
        return json.dumps(error.to_dict(), indent=2)
    
    @staticmethod
    def to_log_message(error: MCPError) -> str:
        """Format error as log message."""
        message = f"[{error.category.value.upper()}] {error.code.value}: {error.message}"
        
        if error.details:
            message += f" - Details: {json.dumps(error.details)}"
            
        if error.cause:
            message += f" - Cause: {str(error.cause)}"
            
        return message
    
    @staticmethod
    def to_user_message(error: MCPError) -> str:
        """Format error as user-friendly message."""
        # Different categories get different user-friendly messages
        category = error.category
        
        if category == ErrorCategory.AUTHENTICATION:
            return "Authentication error: Please check your credentials and try again."
        elif category == ErrorCategory.AUTHORIZATION:
            return "Authorization error: You don't have permission to perform this action."
        elif category == ErrorCategory.VALIDATION:
            return f"Validation error: {error.message}"
        elif category == ErrorCategory.CONNECTION:
            return "Connection error: Unable to connect to the server. Please check your network connection."
        elif category == ErrorCategory.TIMEOUT:
            return "Timeout error: The operation took too long to complete. Please try again later."
        elif category == ErrorCategory.SERVICE:
            return "Service error: The service is temporarily unavailable. Please try again later."
        else:
            return "An unexpected error occurred. Please try again later."


# ===== Usage Examples =====

def demonstrate_validation():
    """Demonstrate validation helpers."""
    # Validate required parameters
    params = {"name": "John", "age": 30}
    missing = ValidationHelper.validate_required(params, ["name", "email", "age"])
    if missing:
        raise ValidationError(
            message=f"Missing required parameters: {', '.join(missing)}",
            code=ErrorCode.VALIDATION_MISSING_PARAMETER,
            details={"missing": missing}
        )
    
    # Validate parameter types
    type_error = ValidationHelper.validate_type(params["age"], int, "age")
    if type_error:
        raise ValidationError(
            message=type_error,
            code=ErrorCode.VALIDATION_INVALID_PARAMETER,
            details={"parameter": "age", "value": params["age"]}
        )
    
    # Validate parameter range
    range_error = ValidationHelper.validate_range(params["age"], 18, 100, "age")
    if range_error:
        raise ValidationError(
            message=range_error,
            code=ErrorCode.VALIDATION_INVALID_PARAMETER,
            details={"parameter": "age", "value": params["age"]}
        )


def demonstrate_error_strategies():
    """Demonstrate error handling strategies."""
    # Example function that might raise an error
    def divide(a, b):
        if b == 0:
            raise ValidationError(
                message="Cannot divide by zero",
                code=ErrorCode.VALIDATION_INVALID_PARAMETER,
                details={"parameter": "b", "value": b}
            )
        return a / b
    
    # Log and raise strategy
    log_and_raise = LogAndRaiseStrategy()
    try:
        log_and_raise.handle(divide, 10, 0)
    except MCPError as e:
        print(f"Log and raise strategy caught: {e}")
    
    # Retry strategy
    retry_strategy = RetryStrategy(max_retries=3, retry_delay=0.1)
    try:
        retry_strategy.handle(divide, 10, 0)
    except MCPError as e:
        print(f"Retry strategy caught: {e}")
    
    # Fallback strategy
    fallback_strategy = FallbackStrategy(fallback_value="N/A")
    result = fallback_strategy.handle(divide, 10, 0)
    print(f"Fallback strategy returned: {result}")


def demonstrate_error_formatting():
    """Demonstrate error formatting."""
    # Create an error
    error = ValidationError(
        message="Parameter 'age' must be at least 18, got 15",
        code=ErrorCode.VALIDATION_INVALID_PARAMETER,
        details={"parameter": "age", "min_value": 18, "actual_value": 15}
    )
    
    # Format as JSON
    json_format = ErrorFormatter.to_json(error)
    print(f"JSON format: {json_format}")
    
    # Format as log message
    log_format = ErrorFormatter.to_log_message(error)
    print(f"Log format: {log_format}")
    
    # Format as user message
    user_format = ErrorFormatter.to_user_message(error)
    print(f"User format: {user_format}")


if __name__ == "__main__":
    print("=== Demonstrating Validation ===")
    try:
        demonstrate_validation()
    except MCPError as e:
        print(f"Validation error: {e}")
    
    print("\n=== Demonstrating Error Strategies ===")
    demonstrate_error_strategies()
    
    print("\n=== Demonstrating Error Formatting ===")
    demonstrate_error_formatting() 