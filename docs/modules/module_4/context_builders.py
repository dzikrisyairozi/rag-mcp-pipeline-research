#!/usr/bin/env python3
"""
MCP Context Builders
===================

This module provides builder patterns for MCP context creation,
with fluent APIs, type-safe validation, factory methods, and
template-based context generation.

Key components:
- ContextBuilder - Generic fluent builder for contexts
- Parameter Validators - Type validation with descriptive errors
- Context Templates - Reusable context templates for common operations
- Factory Methods - Specialized builders for different context types
"""

import re
import json
import uuid
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Type, TypeVar, Generic, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Type definitions
T = TypeVar('T')
P = TypeVar('P')  # Parameter Type


# ===== Validators =====

class ParameterValidator(Generic[P]):
    """Validates parameters for a specific type with descriptive error messages."""
    
    def __init__(self, 
                 name: str, 
                 required: bool = False, 
                 description: str = "",
                 examples: List[str] = None):
        self.name = name
        self.required = required
        self.description = description
        self.examples = examples or []
    
    def validate(self, value: Optional[P]) -> Optional[str]:
        """Validate parameter value, returning error message if invalid."""
        if value is None:
            if self.required:
                return f"Parameter '{self.name}' is required"
            return None
        return self._validate_type(value)
    
    @abstractmethod
    def _validate_type(self, value: P) -> Optional[str]:
        """Validate the type and constraints of the value."""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema representation of the validator."""
        schema = {
            "description": self.description,
            "required": self.required
        }
        if self.examples:
            schema["examples"] = self.examples
        return schema


class StringValidator(ParameterValidator[str]):
    """Validates string parameters."""
    
    def __init__(self, 
                 name: str, 
                 required: bool = False,
                 min_length: int = None,
                 max_length: int = None, 
                 pattern: str = None,
                 description: str = "",
                 examples: List[str] = None):
        super().__init__(name, required, description, examples)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self._pattern_re = re.compile(pattern) if pattern else None
    
    def _validate_type(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return f"Parameter '{self.name}' must be a string"
        
        if self.min_length is not None and len(value) < self.min_length:
            return f"Parameter '{self.name}' must be at least {self.min_length} characters"
        
        if self.max_length is not None and len(value) > self.max_length:
            return f"Parameter '{self.name}' must be at most {self.max_length} characters"
        
        if self._pattern_re and not self._pattern_re.match(value):
            return f"Parameter '{self.name}' must match pattern '{self.pattern}'"
        
        return None
    
    def get_schema(self) -> Dict[str, Any]:
        schema = super().get_schema()
        schema["type"] = "string"
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.pattern:
            schema["pattern"] = self.pattern
        return schema


class IntegerValidator(ParameterValidator[int]):
    """Validates integer parameters."""
    
    def __init__(self, 
                 name: str, 
                 required: bool = False,
                 minimum: int = None,
                 maximum: int = None,
                 description: str = "",
                 examples: List[int] = None):
        super().__init__(name, required, description, examples)
        self.minimum = minimum
        self.maximum = maximum
    
    def _validate_type(self, value: Any) -> Optional[str]:
        if not isinstance(value, int) or isinstance(value, bool):
            return f"Parameter '{self.name}' must be an integer"
        
        if self.minimum is not None and value < self.minimum:
            return f"Parameter '{self.name}' must be at least {self.minimum}"
        
        if self.maximum is not None and value > self.maximum:
            return f"Parameter '{self.name}' must be at most {self.maximum}"
        
        return None
    
    def get_schema(self) -> Dict[str, Any]:
        schema = super().get_schema()
        schema["type"] = "integer"
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        return schema


class BooleanValidator(ParameterValidator[bool]):
    """Validates boolean parameters."""
    
    def _validate_type(self, value: Any) -> Optional[str]:
        if not isinstance(value, bool):
            return f"Parameter '{self.name}' must be a boolean"
        return None
    
    def get_schema(self) -> Dict[str, Any]:
        schema = super().get_schema()
        schema["type"] = "boolean"
        return schema


class DateValidator(ParameterValidator[str]):
    """Validates date parameters in ISO format."""
    
    def _validate_type(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return f"Parameter '{self.name}' must be a string in ISO date format (YYYY-MM-DD)"
        
        try:
            datetime.datetime.strptime(value, "%Y-%m-%d")
            return None
        except ValueError:
            return f"Parameter '{self.name}' must be in ISO date format (YYYY-MM-DD)"
    
    def get_schema(self) -> Dict[str, Any]:
        schema = super().get_schema()
        schema["type"] = "string"
        schema["format"] = "date"
        return schema


class DateTimeValidator(ParameterValidator[str]):
    """Validates datetime parameters in ISO format."""
    
    def _validate_type(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return f"Parameter '{self.name}' must be a string in ISO datetime format"
        
        try:
            datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
            return None
        except ValueError:
            return f"Parameter '{self.name}' must be in ISO datetime format"
    
    def get_schema(self) -> Dict[str, Any]:
        schema = super().get_schema()
        schema["type"] = "string"
        schema["format"] = "date-time"
        return schema


class ObjectValidator(ParameterValidator[Dict[str, Any]]):
    """Validates object parameters with nested schema."""
    
    def __init__(self, 
                 name: str, 
                 required: bool = False,
                 properties: Dict[str, ParameterValidator] = None,
                 description: str = "",
                 examples: List[Dict[str, Any]] = None):
        super().__init__(name, required, description, examples)
        self.properties = properties or {}
    
    def _validate_type(self, value: Any) -> Optional[str]:
        if not isinstance(value, dict):
            return f"Parameter '{self.name}' must be an object"
        
        for prop_name, validator in self.properties.items():
            prop_value = value.get(prop_name)
            error = validator.validate(prop_value)
            if error:
                return f"{error} in object '{self.name}'"
        
        return None
    
    def get_schema(self) -> Dict[str, Any]:
        schema = super().get_schema()
        schema["type"] = "object"
        schema["properties"] = {
            name: validator.get_schema()
            for name, validator in self.properties.items()
        }
        schema["required"] = [
            name for name, validator in self.properties.items()
            if validator.required
        ]
        return schema


class ArrayValidator(ParameterValidator[List[Any]]):
    """Validates array parameters with item validation."""
    
    def __init__(self, 
                 name: str, 
                 required: bool = False,
                 items_validator: ParameterValidator = None,
                 min_items: int = None,
                 max_items: int = None,
                 description: str = "",
                 examples: List[List[Any]] = None):
        super().__init__(name, required, description, examples)
        self.items_validator = items_validator
        self.min_items = min_items
        self.max_items = max_items
    
    def _validate_type(self, value: Any) -> Optional[str]:
        if not isinstance(value, list):
            return f"Parameter '{self.name}' must be an array"
        
        if self.min_items is not None and len(value) < self.min_items:
            return f"Parameter '{self.name}' must have at least {self.min_items} items"
        
        if self.max_items is not None and len(value) > self.max_items:
            return f"Parameter '{self.name}' must have at most {self.max_items} items"
        
        if self.items_validator:
            for i, item in enumerate(value):
                error = self.items_validator.validate(item)
                if error:
                    return f"{error} at index {i} in array '{self.name}'"
        
        return None
    
    def get_schema(self) -> Dict[str, Any]:
        schema = super().get_schema()
        schema["type"] = "array"
        if self.items_validator:
            schema["items"] = self.items_validator.get_schema()
        if self.min_items is not None:
            schema["minItems"] = self.min_items
        if self.max_items is not None:
            schema["maxItems"] = self.max_items
        return schema


# ===== Context Builder =====

class ContextBuilder:
    """
    Fluent builder for creating context requests.
    
    Example:
        builder = ContextBuilder("quickbooks.getInvoices", "accounting")
            .with_parameter("customer_id", "12345")
            .with_parameter("date_range", {"start": "2023-01-01", "end": "2023-12-31"})
            .with_timeout(60)
            .with_priority("high")
        
        context_request = builder.build()
    """
    
    def __init__(self, context_name: str, service_target: str):
        self.context_name = context_name
        self.service_target = service_target
        self.parameters = {}
        self.timeout_seconds = 30
        self.priority = "normal"
        self.idempotency_key = None
        self.validators = {}
    
    def with_parameter(self, name: str, value: Any) -> 'ContextBuilder':
        """Add a parameter to the context."""
        self.parameters[name] = value
        return self
    
    def with_parameters(self, params: Dict[str, Any]) -> 'ContextBuilder':
        """Add multiple parameters to the context."""
        self.parameters.update(params)
        return self
    
    def with_timeout(self, seconds: int) -> 'ContextBuilder':
        """Set the timeout in seconds."""
        self.timeout_seconds = seconds
        return self
    
    def with_priority(self, priority: str) -> 'ContextBuilder':
        """Set the execution priority."""
        self.priority = priority
        return self
    
    def with_idempotency_key(self, key: str = None) -> 'ContextBuilder':
        """Set idempotency key (generates UUID if None)."""
        self.idempotency_key = key or str(uuid.uuid4())
        return self
    
    def with_validator(self, name: str, validator: ParameterValidator) -> 'ContextBuilder':
        """Add a parameter validator."""
        self.validators[name] = validator
        return self
    
    def validate(self) -> List[str]:
        """Validate the context parameters, returning list of errors."""
        errors = []
        
        # Basic validation
        if not self.context_name:
            errors.append("context_name must not be empty")
        
        if not self.service_target:
            errors.append("service_target must not be empty")
        
        if self.timeout_seconds < 1:
            errors.append("timeout_seconds must be at least 1")
        
        # Parameter validation using validators
        for name, validator in self.validators.items():
            value = self.parameters.get(name)
            error = validator.validate(value)
            if error:
                errors.append(error)
        
        return errors
    
    def build(self) -> Dict[str, Any]:
        """Build the context request, raising errors if validation fails."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid context: {'; '.join(errors)}")
        
        request = {
            "context_name": self.context_name,
            "service_target": self.service_target,
            "parameters": self.parameters,
            "timeout_seconds": self.timeout_seconds,
            "priority": self.priority
        }
        
        if self.idempotency_key:
            request["idempotency_key"] = self.idempotency_key
        
        return request
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for the parameters."""
        return {
            "type": "object",
            "properties": {
                name: validator.get_schema()
                for name, validator in self.validators.items()
            },
            "required": [
                name for name, validator in self.validators.items()
                if validator.required
            ]
        }


# ===== Specialized Context Builders =====

class QuickBooksContextBuilder(ContextBuilder):
    """Specialized builder for QuickBooks API contexts."""
    
    def __init__(self, endpoint: str):
        super().__init__(f"quickbooks.{endpoint}", "accounting")
    
    @classmethod
    def get_invoices(cls) -> 'QuickBooksContextBuilder':
        """Factory method for getting invoices."""
        builder = cls("getInvoices")
        
        # Add parameter validators
        builder.with_validator(
            "customer_id",
            StringValidator(
                name="customer_id",
                required=False,
                description="Filter invoices by customer ID"
            )
        )
        
        builder.with_validator(
            "date_range",
            ObjectValidator(
                name="date_range",
                required=False,
                properties={
                    "start": DateValidator(
                        name="start",
                        required=True,
                        description="Start date (YYYY-MM-DD)"
                    ),
                    "end": DateValidator(
                        name="end",
                        required=True,
                        description="End date (YYYY-MM-DD)"
                    )
                },
                description="Date range for filtering invoices"
            )
        )
        
        builder.with_validator(
            "status",
            StringValidator(
                name="status",
                required=False,
                description="Invoice status (paid, unpaid, overdue)",
                pattern="^(paid|unpaid|overdue)$"
            )
        )
        
        return builder
    
    @classmethod
    def create_invoice(cls) -> 'QuickBooksContextBuilder':
        """Factory method for creating an invoice."""
        builder = cls("createInvoice")
        
        # Add parameter validators
        builder.with_validator(
            "customer_id",
            StringValidator(
                name="customer_id",
                required=True,
                description="Customer ID"
            )
        )
        
        builder.with_validator(
            "invoice_date",
            DateValidator(
                name="invoice_date",
                required=True,
                description="Invoice date (YYYY-MM-DD)"
            )
        )
        
        builder.with_validator(
            "due_date",
            DateValidator(
                name="due_date",
                required=True,
                description="Due date (YYYY-MM-DD)"
            )
        )
        
        builder.with_validator(
            "line_items",
            ArrayValidator(
                name="line_items",
                required=True,
                min_items=1,
                items_validator=ObjectValidator(
                    name="line_item",
                    required=True,
                    properties={
                        "description": StringValidator(
                            name="description",
                            required=True,
                            description="Line item description"
                        ),
                        "amount": ObjectValidator(
                            name="amount",
                            required=True,
                            properties={
                                "value": StringValidator(
                                    name="value",
                                    required=True,
                                    description="Monetary value",
                                    pattern="^\\d+(\\.\\d{1,2})?$"
                                ),
                                "currency": StringValidator(
                                    name="currency",
                                    required=True,
                                    description="Currency code (e.g., USD)",
                                    pattern="^[A-Z]{3}$"
                                )
                            }
                        ),
                        "quantity": IntegerValidator(
                            name="quantity",
                            required=True,
                            minimum=1,
                            description="Quantity"
                        ),
                        "tax_code": StringValidator(
                            name="tax_code",
                            required=False,
                            description="Tax code"
                        )
                    }
                ),
                description="Invoice line items"
            )
        )
        
        # Always use idempotency key for invoice creation
        builder.with_idempotency_key()
        
        return builder


class SalesforceContextBuilder(ContextBuilder):
    """Specialized builder for Salesforce API contexts."""
    
    def __init__(self, endpoint: str):
        super().__init__(f"salesforce.{endpoint}", "crm")
    
    @classmethod
    def query_contacts(cls) -> 'SalesforceContextBuilder':
        """Factory method for querying contacts."""
        builder = cls("queryContacts")
        
        # Add parameter validators
        builder.with_validator(
            "query",
            StringValidator(
                name="query",
                required=False,
                description="SOQL query string"
            )
        )
        
        builder.with_validator(
            "filters",
            ObjectValidator(
                name="filters",
                required=False,
                description="Filter criteria"
            )
        )
        
        builder.with_validator(
            "fields",
            ArrayValidator(
                name="fields",
                required=False,
                items_validator=StringValidator(
                    name="field",
                    required=True
                ),
                description="Fields to return"
            )
        )
        
        return builder
    
    @classmethod
    def create_opportunity(cls) -> 'SalesforceContextBuilder':
        """Factory method for creating an opportunity."""
        builder = cls("createOpportunity")
        
        # Add parameter validators
        builder.with_validator(
            "name",
            StringValidator(
                name="name",
                required=True,
                description="Opportunity name"
            )
        )
        
        builder.with_validator(
            "account_id",
            StringValidator(
                name="account_id",
                required=True,
                description="Account ID"
            )
        )
        
        builder.with_validator(
            "stage",
            StringValidator(
                name="stage",
                required=True,
                description="Opportunity stage"
            )
        )
        
        builder.with_validator(
            "close_date",
            DateValidator(
                name="close_date",
                required=True,
                description="Expected close date"
            )
        )
        
        builder.with_validator(
            "amount",
            ObjectValidator(
                name="amount",
                required=False,
                properties={
                    "value": StringValidator(
                        name="value",
                        required=True,
                        description="Monetary value",
                        pattern="^\\d+(\\.\\d{1,2})?$"
                    ),
                    "currency": StringValidator(
                        name="currency",
                        required=True,
                        description="Currency code (e.g., USD)",
                        pattern="^[A-Z]{3}$"
                    )
                }
            )
        )
        
        # Always use idempotency key for opportunity creation
        builder.with_idempotency_key()
        
        return builder


# ===== Context Templates =====

class ContextTemplate:
    """Template for creating contexts with predefined structure."""
    
    def __init__(self, 
                 context_name_template: str,
                 service_target: str,
                 parameter_templates: Dict[str, Any] = None,
                 validators: Dict[str, ParameterValidator] = None):
        self.context_name_template = context_name_template
        self.service_target = service_target
        self.parameter_templates = parameter_templates or {}
        self.validators = validators or {}
    
    def create_builder(self, **kwargs) -> ContextBuilder:
        """Create a builder from this template with provided values."""
        # Format context name with provided values
        context_name = self.context_name_template
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            if placeholder in context_name:
                context_name = context_name.replace(placeholder, str(value))
        
        # Create builder
        builder = ContextBuilder(context_name, self.service_target)
        
        # Add validators
        for name, validator in self.validators.items():
            builder.with_validator(name, validator)
        
        # Process parameter templates
        parameters = {}
        for param_name, template in self.parameter_templates.items():
            if isinstance(template, str):
                # String template with placeholders
                param_value = template
                for key, value in kwargs.items():
                    placeholder = f"{{{key}}}"
                    if placeholder in param_value:
                        param_value = param_value.replace(placeholder, str(value))
                parameters[param_name] = param_value
            elif callable(template):
                # Function template
                try:
                    param_value = template(**{k: v for k, v in kwargs.items() 
                                            if k in template.__code__.co_varnames})
                    parameters[param_name] = param_value
                except Exception as e:
                    raise ValueError(f"Error processing template for parameter '{param_name}': {str(e)}")
            else:
                # Static value
                parameters[param_name] = template
        
        # Add additional parameters from kwargs that weren't used in templates
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            if (key not in parameters and 
                not any(placeholder in name for name in [self.context_name_template] + 
                       list(self.parameter_templates.keys()))):
                parameters[key] = value
        
        # Add parameters to builder
        builder.with_parameters(parameters)
        
        return builder


# ===== Usage Examples =====

def invoice_query_example():
    """Example of using the QuickBooks invoice query builder."""
    # Use the specialized builder
    builder = QuickBooksContextBuilder.get_invoices()
    
    # Set parameters using the fluent API
    builder.with_parameter("customer_id", "12345")
    builder.with_parameter("date_range", {
        "start": "2023-01-01",
        "end": "2023-12-31"
    })
    builder.with_parameter("status", "unpaid")
    
    # Set additional options
    builder.with_timeout(60)
    builder.with_priority("high")
    
    # Build the context request
    try:
        context_request = builder.build()
        print(json.dumps(context_request, indent=2))
        
        # Also show parameter schema
        schema = builder.get_parameter_schema()
        print("\nParameter Schema:")
        print(json.dumps(schema, indent=2))
        
    except ValueError as e:
        print(f"Error: {str(e)}")


def template_example():
    """Example of using context templates."""
    # Create a template for fetching entity by ID
    entity_template = ContextTemplate(
        context_name_template="{service}.get{entity}ById",
        service_target="{service_target}",
        parameter_templates={
            "id": "{entity_id}",
            "include_deleted": False,
            "fields": lambda entity: default_fields_for_entity(entity)
        },
        validators={
            "id": StringValidator(
                name="id",
                required=True,
                description="Entity ID"
            ),
            "include_deleted": BooleanValidator(
                name="include_deleted",
                required=False,
                description="Include deleted entities"
            ),
            "fields": ArrayValidator(
                name="fields",
                required=False,
                items_validator=StringValidator(
                    name="field",
                    required=True
                ),
                description="Fields to include in response"
            )
        }
    )
    
    # Helper function for the template
    def default_fields_for_entity(entity):
        if entity.lower() == "customer":
            return ["id", "name", "email", "phone", "created_at"]
        elif entity.lower() == "invoice":
            return ["id", "number", "customer_id", "total", "date", "due_date", "status"]
        else:
            return ["id", "name"]
    
    # Use the template to create a builder
    builder = entity_template.create_builder(
        service="quickbooks",
        service_target="accounting",
        entity="Customer",
        entity_id="67890"
    )
    
    # Add additional parameters
    builder.with_parameter("include_notes", True)
    
    # Build the context request
    try:
        context_request = builder.build()
        print(json.dumps(context_request, indent=2))
    except ValueError as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # This is just a demonstration - not meant to be run directly
    print("This module demonstrates the MCP context builders.")
    print("Import and use the classes in your own code instead of running this file directly.") 