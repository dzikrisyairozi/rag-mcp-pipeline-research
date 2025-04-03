#!/usr/bin/env python3
"""
ACME Corp Client Integration Example
====================================

This file demonstrates how to use the MCP client SDK to integrate with
various backend systems for ACME Corporation, including:

1. QuickBooks for accounting
2. Salesforce for CRM
3. Custom ERP system

It shows how to set up connections, create contexts, and handle responses.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Import MCP client SDK components
from client_sdk_architecture import (
    MCPClient, 
    ClientConfig, 
    ConnectionManager,
    AuthStrategy,
    TokenProvider
)

# Import context builders
from context_builders import (
    ContextBuilder,
    BatchContextBuilder
)

# Import async patterns
from async_patterns import (
    ContextProcessor,
    ResponseHandler,
    RateLimiter,
    RetryStrategy
)

# Import entity models
from entity_models import (
    Customer,
    Invoice,
    InvoiceItem,
    Money,
    Address,
    EntityStatus
)

# Import integration modules
from accounting_integration import (
    QboConfig,
    QboAuthManager,
    QuickBooksService,
    QuickBooksContextHandler
)

from crm_integration import (
    SalesforceConfig,
    SalesforceAuth,
    SalesforceService,
    SalesforceContextHandler
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('acme_integration')


class AcmeTokenProvider(TokenProvider):
    """Token provider for ACME Corp integrations."""
    
    def __init__(self, config_path: str):
        """Initialize with path to config file."""
        self.config_path = config_path
        self.tokens = {}
        self._load_tokens()
        
    def _load_tokens(self):
        """Load tokens from config file."""
        try:
            with open(self.config_path, 'r') as f:
                self.tokens = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load tokens: {str(e)}")
            self.tokens = {}
    
    def _save_tokens(self):
        """Save tokens to config file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.tokens, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save tokens: {str(e)}")
    
    async def get_token(self, service_name: str) -> Optional[str]:
        """Get token for the specified service."""
        if service_name not in self.tokens:
            return None
            
        token_data = self.tokens[service_name]
        
        # Check if token is expired
        expires_at = datetime.fromisoformat(token_data.get('expires_at', '2000-01-01T00:00:00'))
        if expires_at <= datetime.now():
            logger.info(f"Token for {service_name} is expired, needs refresh")
            return None
            
        return token_data.get('access_token')
    
    async def store_token(self, service_name: str, token: str, expires_in: int = 3600):
        """Store token for the specified service."""
        expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        self.tokens[service_name] = {
            'access_token': token,
            'expires_at': expires_at.isoformat()
        }
        
        self._save_tokens()


class AcmeIntegrationManager:
    """Manages integrations for ACME Corp."""
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize with path to config file."""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up token provider
        self.token_provider = AcmeTokenProvider(config_path)
        
        # Initialize MCP client
        self.client = self._init_mcp_client()
        
        # Set up service handlers
        self.qbo_service = self._init_quickbooks()
        self.sf_service = self._init_salesforce()
        
        # Initialize context handlers
        self.qbo_context_handler = QuickBooksContextHandler(self.qbo_service)
        self.sf_context_handler = SalesforceContextHandler(self.sf_service)
        
        # Set up context processor
        self.context_processor = ContextProcessor(
            handlers=[
                self.qbo_context_handler,
                self.sf_context_handler
            ],
            retry_strategy=RetryStrategy(max_retries=3, delay=1)
        )
    
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
                    "redirect_uri": "https://acmecorp.com/callback",
                    "environment": "sandbox"
                },
                "salesforce": {
                    "client_id": "your_sf_client_id",
                    "client_secret": "your_sf_client_secret",
                    "redirect_uri": "https://acmecorp.com/callback",
                    "environment": "sandbox"
                }
            }
    
    def _init_mcp_client(self) -> MCPClient:
        """Initialize MCP client."""
        mcp_config = self.config.get("mcp_server", {})
        
        client_config = ClientConfig(
            server_url=mcp_config.get("url", "wss://mcp.example.com/ws"),
            api_key=mcp_config.get("api_key", ""),
            client_id="acme_integration",
            connection_timeout=30,
            keep_alive_interval=15
        )
        
        auth_strategy = AuthStrategy.API_KEY
        
        connection_manager = ConnectionManager(
            config=client_config,
            auth_strategy=auth_strategy,
            token_provider=self.token_provider
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
        
        auth_manager = QboAuthManager(
            config=config,
            token_provider=self.token_provider
        )
        
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
        
        auth = SalesforceAuth(
            config=config,
            token_provider=self.token_provider
        )
        
        return SalesforceService(
            config=config,
            auth=auth
        )
    
    async def connect(self):
        """Connect to MCP server and authenticate with services."""
        # Connect to MCP server
        logger.info("Connecting to MCP server...")
        await self.client.connect()
        
        # Authenticate with QuickBooks
        logger.info("Authenticating with QuickBooks...")
        if not await self.qbo_service.auth_manager.is_authenticated():
            auth_url = self.qbo_service.auth_manager.get_authorization_url()
            logger.info(f"Please visit the following URL to authorize QuickBooks access: {auth_url}")
            auth_code = input("Enter the authorization code: ")
            await self.qbo_service.auth_manager.exchange_code_for_token(auth_code)
        
        # Authenticate with Salesforce
        logger.info("Authenticating with Salesforce...")
        if not await self.sf_service.auth.is_authenticated():
            auth_url = self.sf_service.auth.get_authorization_url()
            logger.info(f"Please visit the following URL to authorize Salesforce access: {auth_url}")
            auth_code = input("Enter the authorization code: ")
            await self.sf_service.auth.exchange_code_for_token(auth_code)
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        await self.client.disconnect()
    
    async def sync_customer(self, customer_id: str):
        """Sync customer between QuickBooks and Salesforce."""
        logger.info(f"Syncing customer {customer_id} between QuickBooks and Salesforce")
        
        # Get customer from QuickBooks
        qbo_customer = await self.qbo_service.get_customer(customer_id)
        if not qbo_customer:
            logger.error(f"Customer {customer_id} not found in QuickBooks")
            return
        
        # Create standardized customer entity
        customer = Customer(
            id=customer_id,
            external_id=qbo_customer.get("Id"),
            display_name=qbo_customer.get("DisplayName", ""),
            first_name=qbo_customer.get("GivenName"),
            last_name=qbo_customer.get("FamilyName"),
            company_name=qbo_customer.get("CompanyName"),
            email=qbo_customer.get("PrimaryEmailAddr", {}).get("Address") if qbo_customer.get("PrimaryEmailAddr") else None,
            phone=qbo_customer.get("PrimaryPhone", {}).get("FreeFormNumber") if qbo_customer.get("PrimaryPhone") else None,
            source_system="quickbooks"
        )
        
        # Convert customer to Salesforce format
        sf_contact = {
            "FirstName": customer.first_name or "",
            "LastName": customer.last_name or "(Unknown)",
            "Email": customer.email,
            "Phone": customer.phone,
            "AccountId": None,  # This would need to be mapped to a Salesforce account
            "QuickBooks_ID__c": customer.external_id
        }
        
        # Check if customer exists in Salesforce
        sf_contact_id = await self.sf_service.find_contact_by_email(customer.email) if customer.email else None
        
        if sf_contact_id:
            # Update existing contact
            logger.info(f"Updating Salesforce contact {sf_contact_id}")
            await self.sf_service.update_contact(sf_contact_id, sf_contact)
        else:
            # Create new contact
            logger.info("Creating new Salesforce contact")
            sf_contact_id = await self.sf_service.create_contact(sf_contact)
        
        logger.info(f"Customer sync completed. Salesforce Contact ID: {sf_contact_id}")
    
    async def create_invoice(self, customer_id: str, items: List[Dict[str, Any]]):
        """Create invoice in QuickBooks and notify Salesforce."""
        logger.info(f"Creating invoice for customer {customer_id}")
        
        # Get customer from QuickBooks
        qbo_customer = await self.qbo_service.get_customer(customer_id)
        if not qbo_customer:
            logger.error(f"Customer {customer_id} not found in QuickBooks")
            return
        
        # Create invoice items
        invoice_items = []
        for item_data in items:
            invoice_item = InvoiceItem(
                description=item_data.get("description", ""),
                quantity=float(item_data.get("quantity", 1)),
                unit_price=Money(float(item_data.get("unit_price", 0))),
                item_id=item_data.get("item_id")
            )
            invoice_items.append(invoice_item)
        
        # Create standardized invoice entity
        invoice = Invoice(
            customer_id=customer_id,
            customer_name=qbo_customer.get("DisplayName", ""),
            line_items=invoice_items,
            terms="Net 30",
            notes="Thank you for your business!"
        )
        
        # Validate invoice
        validation_errors = invoice.validate()
        if validation_errors:
            logger.error(f"Invoice validation errors: {validation_errors}")
            return
        
        # Create invoice in QuickBooks
        qbo_invoice = await self.qbo_service.create_invoice(invoice)
        
        # Update invoice with external ID
        invoice.external_id = qbo_invoice.get("Id")
        
        logger.info(f"Invoice created in QuickBooks with ID: {invoice.external_id}")
        
        # Find the corresponding Salesforce opportunity or account
        if invoice.customer_name and "@" in invoice.customer_name:
            sf_contact = await self.sf_service.find_contact_by_email(invoice.customer_name)
            if sf_contact:
                # Create a note on the contact
                note = {
                    "ParentId": sf_contact.get("Id"),
                    "Title": f"Invoice {invoice.doc_number or ''} Created",
                    "Body": f"Invoice created in QuickBooks for {invoice.total.value} {invoice.total.currency}.\n"
                            f"Line items: {', '.join(item.description for item in invoice.line_items)}"
                }
                await self.sf_service.create_note(note)
                logger.info(f"Note created in Salesforce for contact {sf_contact.get('Id')}")
        
        return invoice
    
    async def process_batch_contexts(self, contexts: List[Dict[str, Any]]):
        """Process a batch of contexts using the MCP framework."""
        logger.info(f"Processing batch of {len(contexts)} contexts")
        
        # Create batch context builder
        batch_builder = BatchContextBuilder()
        
        # Add contexts to batch
        for context_data in contexts:
            context_type = context_data.get("type")
            if context_type == "quickbooks.customer.get":
                batch_builder.add_context(
                    context_name="quickbooks.customer.get",
                    parameters={
                        "customer_id": context_data.get("customer_id")
                    }
                )
            elif context_type == "quickbooks.invoice.create":
                batch_builder.add_context(
                    context_name="quickbooks.invoice.create",
                    parameters={
                        "customer_id": context_data.get("customer_id"),
                        "items": context_data.get("items", [])
                    }
                )
            elif context_type == "salesforce.contact.get":
                batch_builder.add_context(
                    context_name="salesforce.contact.get",
                    parameters={
                        "email": context_data.get("email")
                    }
                )
        
        # Build batch context
        batch_context = batch_builder.build()
        
        # Send batch context to MCP server
        response = await self.client.send_context(batch_context)
        
        # Process context with registered handlers
        results = await self.context_processor.process(batch_context, response)
        
        return results


async def main():
    """Run the ACME integration example."""
    # Initialize integration manager
    integration = AcmeIntegrationManager()
    
    # Connect to services
    await integration.connect()
    
    try:
        # Example 1: Sync a customer
        await integration.sync_customer("123456")
        
        # Example 2: Create an invoice
        invoice_items = [
            {
                "description": "Consulting Services",
                "quantity": 10,
                "unit_price": 150.0
            },
            {
                "description": "Software License",
                "quantity": 1,
                "unit_price": 499.99
            }
        ]
        invoice = await integration.create_invoice("123456", invoice_items)
        
        # Example 3: Process batch of contexts
        batch_contexts = [
            {
                "type": "quickbooks.customer.get",
                "customer_id": "123456"
            },
            {
                "type": "salesforce.contact.get",
                "email": "john.doe@acmecorp.com"
            }
        ]
        results = await integration.process_batch_contexts(batch_contexts)
        
        # Print results
        print(json.dumps(results, indent=2))
    finally:
        # Disconnect from services
        await integration.disconnect()


if __name__ == "__main__":
    asyncio.run(main()) 