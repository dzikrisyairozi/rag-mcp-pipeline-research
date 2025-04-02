#!/usr/bin/env python3
"""
MCP Server Integration with LLMs

This script demonstrates how to integrate a simple LLM-powered assistant 
with Multi-Cloud Processing (MCP) servers to enable AI-driven automation.

Features:
- LLM-powered API request generation
- MCP server request handling
- Response parsing and template filling
- Simple AI workflow automation

Uses a streamlined model with minimal dependencies.
"""

import os
import sys
import json
import re
import time
import requests
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create output directory if it doesn't exist
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

# Initialize model and tokenizer
print("Loading model and tokenizer...")
try:
    model_name = "distilgpt2"  # A smaller model for practicality
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"✅ Successfully loaded {model_name}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ==========================================================
# MCP Server Configuration
# ==========================================================

MCP_SERVER_URL = "http://localhost:5000"  # Default MCP server URL

# Map of supported services and their command schemas
SERVICE_SCHEMAS = {
    "quickbooks": {
        "create_invoice": {
            "required_params": ["customer_id", "amount", "description"],
            "optional_params": ["due_date", "line_items"],
            "description": "Create a new invoice in QuickBooks"
        },
        "get_customer": {
            "required_params": ["customer_id"],
            "optional_params": [],
            "description": "Get customer details from QuickBooks"
        }
    },
    "salesforce": {
        "create_lead": {
            "required_params": ["first_name", "last_name", "company", "email"],
            "optional_params": ["phone", "status", "source"],
            "description": "Create a new lead in Salesforce"
        },
        "update_opportunity": {
            "required_params": ["opportunity_id", "stage"],
            "optional_params": ["amount", "close_date", "description"],
            "description": "Update an opportunity in Salesforce"
        }
    },
    "mailchimp": {
        "add_subscriber": {
            "required_params": ["email", "list_id"],
            "optional_params": ["first_name", "last_name", "tags"],
            "description": "Add a subscriber to a Mailchimp list"
        },
        "send_campaign": {
            "required_params": ["campaign_id"],
            "optional_params": ["schedule_time"],
            "description": "Send or schedule a Mailchimp campaign"
        }
    }
}

# ==========================================================
# LLM Request Generator
# ==========================================================

class LLMRequestGenerator:
    """Uses LLM to generate MCP server requests based on natural language input"""
    
    def __init__(self):
        self.temperature = 0.3  # Low temperature for more deterministic outputs
    
    def create_prompt(self, user_request: str) -> str:
        """Create a prompt for the LLM to generate a structured API request"""
        # Get available services and commands for prompt context
        services_info = []
        for service, commands in SERVICE_SCHEMAS.items():
            commands_info = []
            for cmd_name, cmd_info in commands.items():
                params = cmd_info["required_params"] + cmd_info["optional_params"]
                commands_info.append(f"  - {cmd_name}: {cmd_info['description']} (params: {', '.join(params)})")
            services_info.append(f"- {service}:\n" + "\n".join(commands_info))
        
        services_context = "\n".join(services_info)
        
        # Create the prompt
        prompt = f"""
        You are an AI assistant that converts natural language requests into structured API calls.
        
        Available services and commands:
        {services_context}
        
        For the following user request, generate a JSON object with these fields:
        - service: The service to use (quickbooks, salesforce, mailchimp)
        - command: The specific command to execute
        - parameters: A dictionary of parameters required for the command
        
        Only include parameters mentioned in the user request or that can be reasonably inferred.
        Format the output as a valid JSON object that can be parsed programmatically.
        
        User request: "{user_request}"
        
        JSON:
        """
        
        return prompt
    
    def generate_request(self, user_request: str) -> Dict[str, Any]:
        """Generate a structured request from natural language"""
        prompt = self.create_prompt(user_request)
        
        try:
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            input_length = len(inputs["input_ids"][0])
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=input_length + 250,
                    temperature=self.temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated text
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            generated_text = full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
            
            # Extract JSON object
            json_match = re.search(r'({[\s\S]*})', generated_text)
            if json_match:
                json_str = json_match.group(1)
                try:
                    request_data = json.loads(json_str)
                    return request_data
                except json.JSONDecodeError:
                    return {"error": "Generated JSON is invalid"}
            else:
                return {"error": "Could not extract JSON from response"}
            
        except Exception as e:
            return {"error": f"Error generating request: {str(e)}"}
    
    def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the generated request against service schemas"""
        if "error" in request_data:
            return request_data
        
        service = request_data.get("service")
        command = request_data.get("command")
        parameters = request_data.get("parameters", {})
        
        # Check if service exists
        if service not in SERVICE_SCHEMAS:
            return {"error": f"Unknown service: {service}"}
        
        # Check if command exists for the service
        if command not in SERVICE_SCHEMAS[service]:
            return {"error": f"Unknown command '{command}' for service '{service}'"}
        
        # Check required parameters
        command_schema = SERVICE_SCHEMAS[service][command]
        missing_params = []
        for param in command_schema["required_params"]:
            if param not in parameters:
                missing_params.append(param)
        
        if missing_params:
            return {
                "error": f"Missing required parameters for {service}.{command}: {', '.join(missing_params)}",
                "partial_request": request_data
            }
        
        # Filter out unknown parameters
        valid_params = command_schema["required_params"] + command_schema["optional_params"]
        filtered_params = {k: v for k, v in parameters.items() if k in valid_params}
        
        # Create the validated request
        validated_request = {
            "service": service,
            "command": command,
            "parameters": filtered_params
        }
        
        return validated_request

# ==========================================================
# MCP Server Client
# ==========================================================

class MCPClient:
    """Client for interacting with MCP servers"""
    
    def __init__(self, server_url: str = MCP_SERVER_URL):
        self.server_url = server_url
    
    def execute_command(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the MCP server"""
        if "error" in request_data:
            return request_data
        
        try:
            endpoint = f"{self.server_url}/execute"
            
            # In a real implementation, we would actually make the HTTP request
            # For demo purposes, we'll simulate the response
            
            # Simulated response (in production, use the requests library)
            # response = requests.post(endpoint, json=request_data)
            # return response.json()
            
            # Simulate a response based on the request
            return self._simulate_response(request_data)
        
        except Exception as e:
            return {"error": f"Error communicating with MCP server: {str(e)}"}
    
    def _simulate_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a response from the MCP server"""
        service = request_data.get("service")
        command = request_data.get("command")
        params = request_data.get("parameters", {})
        
        # Simulate some basic responses
        if service == "quickbooks":
            if command == "create_invoice":
                return {
                    "success": True,
                    "invoice_id": f"INV-{int(time.time())}",
                    "customer_id": params.get("customer_id"),
                    "amount": params.get("amount"),
                    "status": "CREATED"
                }
        elif service == "salesforce":
            if command == "create_lead":
                return {
                    "success": True,
                    "lead_id": f"LEAD-{int(time.time())}",
                    "name": f"{params.get('first_name', '')} {params.get('last_name', '')}",
                    "status": "NEW"
                }
        elif service == "mailchimp":
            if command == "add_subscriber":
                return {
                    "success": True,
                    "email": params.get("email"),
                    "list_id": params.get("list_id"),
                    "status": "SUBSCRIBED"
                }
        
        # Generic success response for other commands
        return {
            "success": True,
            "command": command,
            "service": service,
            "timestamp": time.time()
        }

# ==========================================================
# Response Handler
# ==========================================================

class ResponseHandler:
    """Process and format responses for the user"""
    
    def __init__(self):
        pass
    
    def format_response(self, response: Dict[str, Any], original_request: str) -> str:
        """Format the response from the MCP server into a user-friendly message"""
        if "error" in response:
            return f"Error: {response['error']}"
        
        if response.get("success", False):
            service = response.get("service", "unknown")
            
            # Handle QuickBooks responses
            if service == "quickbooks" or "invoice_id" in response:
                return f"""
                ✅ Invoice successfully created!
                
                Invoice ID: {response.get('invoice_id')}
                Amount: ${response.get('amount')}
                Customer ID: {response.get('customer_id')}
                Status: {response.get('status')}
                
                The invoice has been recorded in QuickBooks and is ready for review.
                """
            
            # Handle Salesforce responses
            elif service == "salesforce" or "lead_id" in response:
                return f"""
                ✅ Lead successfully created in Salesforce!
                
                Lead ID: {response.get('lead_id')}
                Name: {response.get('name')}
                Status: {response.get('status')}
                
                The lead has been added to Salesforce and is ready for follow-up.
                """
            
            # Handle Mailchimp responses
            elif service == "mailchimp" or "email" in response:
                return f"""
                ✅ Subscriber successfully added to Mailchimp!
                
                Email: {response.get('email')}
                List ID: {response.get('list_id')}
                Status: {response.get('status')}
                
                The subscriber has been added to your mailing list.
                """
            
            # Generic success response
            else:
                return f"""
                ✅ Command executed successfully!
                
                Service: {response.get('service', 'Unknown')}
                Command: {response.get('command', 'Unknown')}
                
                Your request has been processed successfully.
                """
        else:
            return f"""
            ❌ Command execution failed.
            
            Service: {response.get('service', 'Unknown')}
            Command: {response.get('command', 'Unknown')}
            Error: {response.get('message', 'Unknown error')}
            
            Please check your request and try again.
            """

# ==========================================================
# AI Automation Workflow
# ==========================================================

class LLM_MCP_Workflow:
    """Main workflow integrating LLM with MCP server"""
    
    def __init__(self, server_url: str = MCP_SERVER_URL):
        self.request_generator = LLMRequestGenerator()
        self.mcp_client = MCPClient(server_url)
        self.response_handler = ResponseHandler()
    
    def process_request(self, user_request: str) -> Dict[str, Any]:
        """Process a natural language request end-to-end"""
        # Step 1: Generate structured request using LLM
        generated_request = self.request_generator.generate_request(user_request)
        
        # Step 2: Validate the request
        validated_request = self.request_generator.validate_request(generated_request)
        
        # Step 3: Send to MCP server if valid
        if "error" not in validated_request:
            response = self.mcp_client.execute_command(validated_request)
        else:
            response = validated_request
        
        # Step 4: Format the response
        formatted_response = self.response_handler.format_response(response, user_request)
        
        return {
            "user_request": user_request,
            "generated_request": generated_request,
            "validated_request": validated_request,
            "raw_response": response,
            "formatted_response": formatted_response
        }

# ==========================================================
# Gradio Interface
# ==========================================================

def create_interface():
    """Create a Gradio interface for the LLM-MCP workflow"""
    workflow = LLM_MCP_Workflow()
    
    with gr.Blocks(title="LLM-MCP Integration") as interface:
        gr.Markdown("# AI-Powered API Automation with MCP")
        gr.Markdown("""
        This demo shows how an LLM can be used to convert natural language requests
        into structured API calls for Multi-Cloud Processing (MCP) servers.
        
        Try asking the system to perform tasks like:
        - "Create an invoice for customer ABC123 for $500 for consulting services"
        - "Add john.doe@example.com to the newsletter list ABC123 in Mailchimp"
        - "Create a lead in Salesforce for John Doe from Acme Corp with email john@acme.com"
        """)
        
        with gr.Row():
            server_url = gr.Textbox(
                label="MCP Server URL",
                value=MCP_SERVER_URL,
                placeholder="http://localhost:5000"
            )
        
        user_request = gr.Textbox(
            label="Your Request (in natural language)",
            placeholder="Create an invoice for customer ABC123 for $500 for consulting services",
            lines=3
        )
        
        submit_button = gr.Button("Process Request")
        
        with gr.Accordion("Generated API Request", open=True):
            request_json = gr.JSON(label="Structured Request")
        
        response_output = gr.Textbox(
            label="Response",
            lines=8,
            interactive=False
        )
        
        with gr.Accordion("Raw Response Data", open=False):
            raw_response = gr.JSON(label="Raw Response")
        
        # Function to process requests
        def process_user_request(request_text, server):
            if not request_text.strip():
                return "{}", "{}", "Please enter a request."
            
            # Update server URL if changed
            workflow.mcp_client.server_url = server
            
            # Process the request
            result = workflow.process_request(request_text)
            
            return (
                result.get("validated_request", {}),
                result.get("raw_response", {}),
                result.get("formatted_response", "Error processing request.")
            )
        
        # Connect the components
        submit_button.click(
            fn=process_user_request,
            inputs=[user_request, server_url],
            outputs=[request_json, raw_response, response_output]
        )
    
    return interface

# ==========================================================
# Main Execution
# ==========================================================

if __name__ == "__main__":
    print("Starting LLM-MCP Integration Demo...")
    print(f"Outputs will be saved to: {output_dir}")
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch(share=False)  # Set to True to create a public link
    
    print("\nLLM-MCP Integration demo is running!")
    print("This completes Module 1. Next up: Module 2 for RAG implementation.") 