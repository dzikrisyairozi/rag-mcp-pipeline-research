#!/usr/bin/env python3
"""
MCP Client Interface Example
===========================

This module demonstrates how to build a client interface that uses
the MCP Client SDK to interact with various backend systems through
a unified API.

Key components:
- MCPClientInterface: Main interface class for interacting with MCP
- TemplateEngine: Simple template engine for generating contexts
- ClientApplication: Example application using the interface
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, date, timedelta

# Import MCP client SDK components
from client_sdk_architecture import (
    MCPClient, 
    ClientConfig, 
    ConnectionManager,
    AuthStrategy,
    Context,
    ContextResult
)

# Import from our integrations
from accounting_integration import QboConfig, QboAuthManager, QuickBooksService
from crm_integration import SalesforceConfig, SalesforceAuth, SalesforceService

# Import entity models
from entity_models import Customer, Invoice, InvoiceItem, Money, Address

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('client_interface')


class TemplateEngine:
    """Simple template engine for generating contexts."""
    
    def __init__(self, templates_path: str = "templates"):
        """Initialize template engine with path to templates."""
        self.templates_path = templates_path
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load context templates from templates directory."""
        templates = {}
        
        # In a real application, this would load templates from files
        # For demonstration, we'll define them inline
        
        templates["quickbooks.customer.get"] = {
            "name": "quickbooks.customer.get",
            "parameters": {
                "customer_id": "${customer_id}"
            }
        }
        
        templates["quickbooks.invoice.create"] = {
            "name": "quickbooks.invoice.create",
            "parameters": {
                "customer_id": "${customer_id}",
                "customer_name": "${customer_name}",
                "doc_number": "${doc_number}",
                "items": "${items}",
                "notes": "${notes}",
                "terms": "${terms}"
            }
        }
        
        templates["salesforce.contact.get"] = {
            "name": "salesforce.contact.get",
            "parameters": {
                "email": "${email}"
            }
        }
        
        templates["salesforce.opportunity.create"] = {
            "name": "salesforce.opportunity.create",
            "parameters": {
                "Name": "${name}",
                "AccountId": "${account_id}",
                "StageName": "${stage_name}",
                "CloseDate": "${close_date}",
                "Amount": "${amount}"
            }
        }
        
        return templates
    
    def render_context(self, template_name: str, values: Dict[str, Any]) -> Context:
        """Render a context template with the provided values."""
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")
            
        template = self.templates[template_name]
        
        # Create a copy of the template
        rendered = {
            "name": template["name"],
            "parameters": {}
        }
        
        # Replace placeholders in parameters
        for key, value in template["parameters"].items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                placeholder = value[2:-1]
                if placeholder in values:
                    rendered["parameters"][key] = values[placeholder]
                else:
                    rendered["parameters"][key] = None
            else:
                rendered["parameters"][key] = value
        
        # Create Context object
        return Context(
            name=rendered["name"],
            parameters=rendered["parameters"]
        )


class MCPClientInterface:
    """Main interface class for interacting with MCP."""
    
    def __init__(
        self,
        config_path: str = "config.json",
        templates_path: str = "templates"
    ):
        """Initialize MCP client interface."""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize MCP client
        self.client = self._init_mcp_client()
        
        # Initialize template engine
        self.template_engine = TemplateEngine(templates_path)
        
        # Set up service clients
        self.qbo_client = self._init_quickbooks()
        self.sf_client = self._init_salesforce()
        
        # Set up context result handlers
        self.result_handlers = {
            "quickbooks.customer.get": self._handle_qbo_customer_result,
            "quickbooks.invoice.create": self._handle_qbo_invoice_result,
            "salesforce.contact.get": self._handle_sf_contact_result,
            "salesforce.opportunity.create": self._handle_sf_opportunity_result
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load config: {str(e)}")
            return {
                "mcp_server": {
                    "url": "wss://mcp.example.com/ws",
                    "api_key": "your_api_key"
                },
                "quickbooks": {
                    "client_id": "your_qbo_client_id",
                    "client_secret": "your_qbo_client_secret",
                    "redirect_uri": "https://your-app.example.com/callback",
                    "environment": "sandbox"
                },
                "salesforce": {
                    "client_id": "your_sf_client_id",
                    "client_secret": "your_sf_client_secret",
                    "redirect_uri": "https://your-app.example.com/callback",
                    "environment": "sandbox"
                }
            }
    
    def _init_mcp_client(self) -> MCPClient:
        """Initialize MCP client."""
        mcp_config = self.config.get("mcp_server", {})
        
        client_config = ClientConfig(
            server_url=mcp_config.get("url", "wss://mcp.example.com/ws"),
            api_key=mcp_config.get("api_key", ""),
            client_id="mcp_client_interface",
            connection_timeout=30,
            keep_alive_interval=15
        )
        
        auth_strategy = AuthStrategy.API_KEY
        
        connection_manager = ConnectionManager(
            config=client_config,
            auth_strategy=auth_strategy
        )
        
        return MCPClient(
            config=client_config,
            connection_manager=connection_manager
        )
    
    def _init_quickbooks(self) -> QuickBooksService:
        """Initialize QuickBooks service."""
        qbo_config = self.config.get("quickbooks", {})
        
        config = QboConfig(
            client_id=qbo_config.get("client_id", ""),
            client_secret=qbo_config.get("client_secret", ""),
            redirect_uri=qbo_config.get("redirect_uri", ""),
            environment=qbo_config.get("environment", "sandbox")
        )
        
        auth_manager = QboAuthManager(config)
        
        return QuickBooksService(
            config=config,
            auth_manager=auth_manager
        )
    
    def _init_salesforce(self) -> SalesforceService:
        """Initialize Salesforce service."""
        sf_config = self.config.get("salesforce", {})
        
        config = SalesforceConfig(
            client_id=sf_config.get("client_id", ""),
            client_secret=sf_config.get("client_secret", ""),
            redirect_uri=sf_config.get("redirect_uri", ""),
            environment=sf_config.get("environment", "sandbox")
        )
        
        auth = SalesforceAuth(config)
        
        return SalesforceService(
            config=config,
            auth=auth
        )
    
    async def connect(self):
        """Connect to MCP server."""
        logger.info("Connecting to MCP server...")
        await self.client.connect()
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        logger.info("Disconnecting from MCP server...")
        await self.client.disconnect()
    
    async def _handle_qbo_customer_result(self, result: ContextResult) -> Any:
        """Handle result from quickbooks.customer.get context."""
        if not result.success:
            logger.error(f"Error getting customer: {result.error}")
            return None
            
        # Convert to Customer entity
        qbo_customer = result.data
        
        customer = Customer(
            external_id=qbo_customer.get("Id"),
            display_name=qbo_customer.get("DisplayName", ""),
            first_name=qbo_customer.get("GivenName"),
            last_name=qbo_customer.get("FamilyName"),
            company_name=qbo_customer.get("CompanyName"),
            email=qbo_customer.get("PrimaryEmailAddr", {}).get("Address") if qbo_customer.get("PrimaryEmailAddr") else None,
            phone=qbo_customer.get("PrimaryPhone", {}).get("FreeFormNumber") if qbo_customer.get("PrimaryPhone") else None,
            source_system="quickbooks"
        )
        
        # Add billing address if available
        if qbo_customer.get("BillAddr"):
            bill_addr = qbo_customer.get("BillAddr")
            customer.billing_address = Address(
                line1=bill_addr.get("Line1"),
                city=bill_addr.get("City"),
                state=bill_addr.get("CountrySubDivisionCode"),
                postal_code=bill_addr.get("PostalCode"),
                country=bill_addr.get("Country")
            )
        
        return customer
    
    async def _handle_qbo_invoice_result(self, result: ContextResult) -> Any:
        """Handle result from quickbooks.invoice.create context."""
        if not result.success:
            logger.error(f"Error creating invoice: {result.error}")
            return None
            
        # Return invoice data as is
        return result.data
    
    async def _handle_sf_contact_result(self, result: ContextResult) -> Any:
        """Handle result from salesforce.contact.get context."""
        if not result.success:
            logger.error(f"Error getting contact: {result.error}")
            return None
            
        # Return contact data as is
        return result.data
    
    async def _handle_sf_opportunity_result(self, result: ContextResult) -> Any:
        """Handle result from salesforce.opportunity.create context."""
        if not result.success:
            logger.error(f"Error creating opportunity: {result.error}")
            return None
            
        # Return opportunity data as is
        return result.data
    
    async def execute_context(self, context: Context) -> Any:
        """Execute a single context and process the result."""
        # Send context to MCP server
        logger.info(f"Executing context: {context.name}")
        result = await self.client.send_context(context)
        
        # Process result with appropriate handler
        handler = self.result_handlers.get(context.name)
        if handler:
            return await handler(result)
        else:
            # Default handling
            if result.success:
                return result.data
            else:
                logger.error(f"Error executing context {context.name}: {result.error}")
                return None
    
    async def execute_template(self, template_name: str, values: Dict[str, Any]) -> Any:
        """Execute a context template with the provided values."""
        # Render context from template
        context = self.template_engine.render_context(template_name, values)
        
        # Execute context
        return await self.execute_context(context)
    
    async def get_customer(self, customer_id: str) -> Optional[Customer]:
        """Get a customer by ID."""
        return await self.execute_template("quickbooks.customer.get", {
            "customer_id": customer_id
        })
    
    async def create_invoice(
        self,
        customer_id: str,
        items: List[Dict[str, Any]],
        doc_number: Optional[str] = None,
        notes: Optional[str] = None,
        terms: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create an invoice."""
        # Get customer info to get the name
        customer = await self.get_customer(customer_id)
        if not customer:
            logger.error(f"Customer not found: {customer_id}")
            return None
        
        return await self.execute_template("quickbooks.invoice.create", {
            "customer_id": customer_id,
            "customer_name": customer.display_name,
            "doc_number": doc_number,
            "items": items,
            "notes": notes,
            "terms": terms or "Net 30"
        })
    
    async def get_contact_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get a Salesforce contact by email."""
        return await self.execute_template("salesforce.contact.get", {
            "email": email
        })
    
    async def create_opportunity(
        self,
        name: str,
        account_id: str,
        stage_name: str,
        amount: float,
        close_date: Optional[date] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a Salesforce opportunity."""
        # If close date not provided, default to 30 days from now
        if not close_date:
            close_date = date.today() + timedelta(days=30)
        
        return await self.execute_template("salesforce.opportunity.create", {
            "name": name,
            "account_id": account_id,
            "stage_name": stage_name,
            "amount": amount,
            "close_date": close_date.isoformat()
        })


class ClientApplication:
    """Example application using the MCP client interface."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize client application."""
        self.interface = MCPClientInterface(config_path)
    
    async def initialize(self):
        """Initialize the application."""
        # Connect to MCP server
        await self.interface.connect()
    
    async def shutdown(self):
        """Shut down the application."""
        # Disconnect from MCP server
        await self.interface.disconnect()
    
    async def create_invoice_from_template(self, template_id: str, customer_id: str) -> Optional[Dict[str, Any]]:
        """Create an invoice from a template."""
        # In a real application, this would load a template from a database
        # For demonstration, we'll define a few templates inline
        templates = {
            "consulting": {
                "doc_number": None,  # Auto-generate
                "items": [
                    {
                        "description": "Consulting Services",
                        "quantity": 10,
                        "unit_price": 150.0
                    },
                    {
                        "description": "Project Management",
                        "quantity": 5,
                        "unit_price": 125.0
                    }
                ],
                "notes": "Thank you for your business!",
                "terms": "Net 30"
            },
            "software": {
                "doc_number": None,  # Auto-generate
                "items": [
                    {
                        "description": "Software License",
                        "quantity": 1,
                        "unit_price": 999.99
                    },
                    {
                        "description": "Implementation Services",
                        "quantity": 20,
                        "unit_price": 175.0
                    },
                    {
                        "description": "Training",
                        "quantity": 8,
                        "unit_price": 125.0
                    }
                ],
                "notes": "License valid for one year from purchase date.",
                "terms": "Net 30"
            }
        }
        
        if template_id not in templates:
            logger.error(f"Invoice template not found: {template_id}")
            return None
        
        template = templates[template_id]
        
        # Create invoice using the template
        return await self.interface.create_invoice(
            customer_id=customer_id,
            items=template["items"],
            doc_number=template["doc_number"],
            notes=template["notes"],
            terms=template["terms"]
        )
    
    async def sync_customer_to_salesforce(self, customer_id: str) -> bool:
        """Sync a customer from QuickBooks to Salesforce."""
        # Get customer from QuickBooks
        customer = await self.interface.get_customer(customer_id)
        if not customer:
            logger.error(f"Customer not found: {customer_id}")
            return False
        
        # Check if contact exists in Salesforce
        sf_contact = None
        if customer.email:
            sf_contact = await self.interface.get_contact_by_email(customer.email)
        
        if sf_contact:
            logger.info(f"Contact already exists in Salesforce: {sf_contact.get('Id')}")
            # In a real application, you might want to update the contact here
            return True
        
        # Create contact in Salesforce
        contact_data = {
            "FirstName": customer.first_name or "",
            "LastName": customer.last_name or "(Unknown)",
            "Email": customer.email,
            "Phone": customer.phone,
            "QuickBooks_ID__c": customer.external_id
        }
        
        # In a real application, you would create the contact here
        logger.info(f"Would create contact in Salesforce: {json.dumps(contact_data)}")
        
        return True
    
    async def run_demo(self):
        """Run a demonstration of the application."""
        try:
            # Initialize the application
            await self.initialize()
            
            # Get a customer
            customer_id = "123456"
            customer = await self.interface.get_customer(customer_id)
            
            if customer:
                logger.info(f"Retrieved customer: {customer.display_name}")
                
                # Create an invoice from a template
                invoice = await self.create_invoice_from_template("consulting", customer_id)
                
                if invoice:
                    logger.info(f"Created invoice with ID: {invoice.get('Id')}")
                    
                    # Sync customer to Salesforce
                    sync_success = await self.sync_customer_to_salesforce(customer_id)
                    
                    if sync_success:
                        logger.info("Successfully synced customer to Salesforce")
                    else:
                        logger.error("Failed to sync customer to Salesforce")
                else:
                    logger.error("Failed to create invoice")
            else:
                logger.error(f"Customer not found: {customer_id}")
        finally:
            # Shut down the application
            await self.shutdown()


async def main():
    """Run the client interface example."""
    app = ClientApplication()
    await app.run_demo()


if __name__ == "__main__":
    asyncio.run(main()) 