#!/usr/bin/env python3
"""
Simple MCP (Multi-Cloud Processing) Server Example

This script demonstrates a basic implementation of an MCP server that can route
commands to different API services. This is a simplified educational example
to help understand the concept of MCP servers.

In a real-world scenario, an MCP server would be more robust, handle authentication,
support many services, and include error handling and logging.
"""

import json
from fastapi import FastAPI, HTTPException, Request
import uvicorn

# Create a FastAPI application
app = FastAPI(title="Simple MCP Server Example")

# Simulated API services (in a real system, these would connect to actual APIs)
SERVICES = {
    "quickbooks": {
        "create_invoice": lambda data: {"invoice_id": "INV-123", "status": "created", "data": data},
        "get_customer": lambda customer_id: {"customer_id": customer_id, "name": "Example Customer", "email": "customer@example.com"},
    },
    "salesforce": {
        "create_lead": lambda data: {"lead_id": "LEAD-456", "status": "created", "data": data},
        "update_opportunity": lambda opp_id, data: {"opportunity_id": opp_id, "status": "updated", "data": data},
    },
    "mailchimp": {
        "add_subscriber": lambda email, list_id: {"result": "success", "email": email, "list_id": list_id},
        "send_campaign": lambda campaign_id: {"campaign_id": campaign_id, "status": "sent"},
    }
}

@app.get("/")
async def root():
    """Root endpoint with basic server information."""
    return {
        "server": "Simple MCP Server Example",
        "version": "0.1.0",
        "description": "A basic demonstration of MCP server concepts",
        "available_services": list(SERVICES.keys()),
        "usage": "POST to /execute with JSON payload containing service, command, and parameters"
    }

@app.post("/execute")
async def execute_command(request: Request):
    """
    Execute a command on a specified service.
    
    The request body should be a JSON object with the following structure:
    {
        "service": "service_name",
        "command": "command_name",
        "parameters": {...} # Parameters for the command
    }
    """
    try:
        # Parse the request body
        data = await request.json()
        
        # Extract service, command, and parameters
        service_name = data.get("service")
        command_name = data.get("command")
        parameters = data.get("parameters", {})
        
        # Validate the request
        if not service_name:
            raise HTTPException(status_code=400, detail="Missing 'service' field")
        if not command_name:
            raise HTTPException(status_code=400, detail="Missing 'command' field")
        
        # Check if the service exists
        if service_name not in SERVICES:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        # Get the service
        service = SERVICES[service_name]
        
        # Check if the command exists in the service
        if command_name not in service:
            raise HTTPException(status_code=404, detail=f"Command '{command_name}' not found in service '{service_name}'")
        
        # Get the command function
        command_func = service[command_name]
        
        # Execute the command with the provided parameters
        if isinstance(parameters, dict):
            result = command_func(**parameters)
        elif isinstance(parameters, list):
            result = command_func(*parameters)
        else:
            result = command_func(parameters)
        
        # Return the result
        return {
            "service": service_name,
            "command": command_name,
            "status": "success",
            "result": result
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle any other exceptions
        raise HTTPException(status_code=500, detail=f"Error executing command: {str(e)}")

# Example usage instructions
USAGE_EXAMPLES = [
    {
        "description": "Create an invoice in QuickBooks",
        "request": {
            "service": "quickbooks",
            "command": "create_invoice",
            "parameters": {
                "data": {
                    "customer_id": "CUST-123",
                    "amount": 100.00,
                    "date": "2023-12-31"
                }
            }
        }
    },
    {
        "description": "Add a subscriber to a Mailchimp list",
        "request": {
            "service": "mailchimp",
            "command": "add_subscriber",
            "parameters": {
                "email": "subscriber@example.com",
                "list_id": "LIST-789"
            }
        }
    }
]

@app.get("/examples")
async def get_examples():
    """Return example usage of the MCP server."""
    return USAGE_EXAMPLES

if __name__ == "__main__":
    print("Starting Simple MCP Server Example...")
    print("This is a demonstration only and doesn't connect to real services.")
    print("\nTo use this server:")
    print("1. Make POST requests to /execute with JSON payloads")
    print("2. Visit /examples for sample requests")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Start the server
    uvicorn.run(app, host="127.0.0.1", port=8000) 