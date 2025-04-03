#!/usr/bin/env python3
"""
MCP QuickBooks Accounting Integration
===================================

This module provides integration with QuickBooks Online through the MCP framework.
It implements authentication, data mapping, and API operations for QuickBooks.

Key components:
- QboConfig: Configuration for QuickBooks API connections
- QboAuthManager: Authentication manager for QuickBooks OAuth flow
- QuickBooksService: Service for interacting with QuickBooks API
- QuickBooksContextHandler: Handler for processing MCP contexts for QuickBooks
"""

import os
import logging
import urllib.parse
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Import MCP client SDK components
from client_sdk_architecture import (
    TokenProvider,
    ContextHandler,
    Context,
    ContextResult
)

# Import entity models
from entity_models import (
    Customer,
    Invoice,
    InvoiceItem,
    Money,
    Address
)

# Try to import QuickBooks libraries, with graceful fallback for documentation
try:
    from intuitlib.client import AuthClient
    from intuitlib.enums import Scopes
    from quickbooks import QuickBooks
    from quickbooks.objects import Customer as QboCustomer
    from quickbooks.objects import Invoice as QboInvoice
    QUICKBOOKS_SDK_AVAILABLE = True
except ImportError:
    QUICKBOOKS_SDK_AVAILABLE = False
    # Create placeholders for documentation purposes
    class AuthClient:
        pass
    class Scopes:
        ACCOUNTING = "com.intuit.quickbooks.accounting"
    class QuickBooks:
        pass
    class QboCustomer:
        pass
    class QboInvoice:
        pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quickbooks_integration')


class QboEnvironment:
    """QuickBooks environment types."""
    PRODUCTION = "production"
    SANDBOX = "sandbox"


class QboConfig:
    """Configuration for QuickBooks API connections."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        environment: str = QboEnvironment.SANDBOX,
        scope: List[str] = None
    ):
        """Initialize QuickBooks configuration."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.environment = environment
        self.scope = scope or [Scopes.ACCOUNTING]
        self.realm_id = None  # Set after authentication
        
    @property
    def is_production(self) -> bool:
        """Check if environment is production."""
        return self.environment == QboEnvironment.PRODUCTION


class QboAuthManager:
    """Authentication manager for QuickBooks OAuth flow."""
    
    def __init__(
        self,
        config: QboConfig,
        token_provider: Optional[TokenProvider] = None
    ):
        """Initialize QuickBooks authentication manager."""
        self.config = config
        self.token_provider = token_provider
        self.access_token = None
        self.refresh_token = None
        self.expires_at = None
        self._auth_client = None
        
    @property
    def auth_client(self) -> AuthClient:
        """Get the Intuit AuthClient instance."""
        if not QUICKBOOKS_SDK_AVAILABLE:
            raise RuntimeError("intuitlib and quickbooks libraries are not installed")
            
        if not self._auth_client:
            self._auth_client = AuthClient(
                client_id=self.config.client_id,
                client_secret=self.config.client_secret,
                redirect_uri=self.config.redirect_uri,
                environment='production' if self.config.is_production else 'sandbox'
            )
        
        return self._auth_client
        
    def get_authorization_url(self) -> str:
        """Get the authorization URL for the OAuth flow."""
        return self.auth_client.get_authorization_url(self.config.scope)
    
    async def exchange_code_for_token(self, code: str, realm_id: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        # In a real application, this would call the Intuit API
        # For documentation purposes, we'll simulate the response
        
        # Simulate token response
        token_data = {
            "access_token": "SIMULATED_ACCESS_TOKEN",
            "refresh_token": "SIMULATED_REFRESH_TOKEN",
            "expires_in": 3600
        }
        
        # Store realm ID
        self.config.realm_id = realm_id
        
        # Update token data
        await self._update_token_data(token_data)
        
        return token_data
    
    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh the access token using the refresh token."""
        # In a real application, this would call the Intuit API
        # For documentation purposes, we'll simulate the response
        
        # Simulate token response
        token_data = {
            "access_token": "SIMULATED_REFRESHED_ACCESS_TOKEN",
            "refresh_token": "SIMULATED_REFRESH_TOKEN",
            "expires_in": 3600
        }
        
        # Update token data
        await self._update_token_data(token_data)
        
        return token_data
    
    async def _update_token_data(self, token_data: Dict[str, Any]):
        """Update token data from response."""
        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token", self.refresh_token)
        
        expires_in = token_data.get("expires_in", 3600)
        self.expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        # Store token if provider is available
        if self.token_provider:
            await self.token_provider.store_token(
                service_name="quickbooks",
                token=self.access_token,
                expires_in=expires_in
            )
    
    async def ensure_token(self) -> str:
        """Ensure a valid access token is available."""
        # If token provider is available, try to get token from it
        if self.token_provider and not self.access_token:
            self.access_token = await self.token_provider.get_token("quickbooks")
            
        # If token is expired or not available, refresh it
        if not self.access_token or (self.expires_at and datetime.now() >= self.expires_at):
            if self.refresh_token:
                await self.refresh_access_token()
            else:
                raise ValueError("No access token or refresh token available")
                
        return self.access_token
    
    async def is_authenticated(self) -> bool:
        """Check if authentication is currently valid."""
        try:
            await self.ensure_token()
            return bool(self.config.realm_id)
        except Exception as e:
            logger.error(f"Error checking authentication: {str(e)}")
            return False


class QuickBooksService:
    """Service for interacting with QuickBooks API."""
    
    def __init__(
        self,
        config: QboConfig,
        auth_manager: QboAuthManager
    ):
        """Initialize QuickBooks service."""
        self.config = config
        self.auth_manager = auth_manager
        self._client = None
        
    async def _get_client(self) -> QuickBooks:
        """Get an authenticated QuickBooks client."""
        if not QUICKBOOKS_SDK_AVAILABLE:
            raise RuntimeError("quickbooks library is not installed")
            
        if not self._client:
            # Get access token
            access_token = await self.auth_manager.ensure_token()
            
            # Create QuickBooks client
            self._client = QuickBooks(
                auth_client=self.auth_manager.auth_client,
                refresh_token=self.auth_manager.refresh_token,
                company_id=self.config.realm_id
            )
            
            # Set access token
            self._client.auth_client.access_token = access_token
            
        return self._client
    
    async def get_company_info(self) -> Dict[str, Any]:
        """Get company information."""
        # In a real application, this would query the QuickBooks API
        # For documentation purposes, we'll simulate the response
        
        return {
            "Id": "1234567890",
            "CompanyName": "ACME Corporation",
            "LegalName": "ACME Corporation LLC",
            "CompanyAddr": {
                "Line1": "123 Main St",
                "City": "San Francisco",
                "CountrySubDivisionCode": "CA",
                "PostalCode": "94105",
                "Country": "US"
            },
            "Email": {
                "Address": "info@acme.example.com"
            },
            "WebAddr": {
                "URI": "https://acme.example.com"
            }
        }
    
    async def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get a customer by ID."""
        # In a real application, this would query the QuickBooks API
        # For documentation purposes, we'll simulate the response
        
        return {
            "Id": customer_id,
            "DisplayName": "ACME Corporation",
            "GivenName": "John",
            "FamilyName": "Doe",
            "CompanyName": "ACME Corporation",
            "PrimaryEmailAddr": {
                "Address": "john.doe@acme.example.com"
            },
            "PrimaryPhone": {
                "FreeFormNumber": "555-123-4567"
            },
            "BillAddr": {
                "Line1": "123 Main St",
                "City": "San Francisco",
                "CountrySubDivisionCode": "CA",
                "PostalCode": "94105",
                "Country": "US"
            }
        }
    
    async def create_customer(self, customer: Customer) -> Dict[str, Any]:
        """Create a new customer."""
        # In a real application, this would call the QuickBooks API
        # For documentation purposes, we'll simulate the response
        
        return {
            "Id": "123456789",
            "SyncToken": "0",
            "DisplayName": customer.display_name,
            "GivenName": customer.first_name,
            "FamilyName": customer.last_name,
            "CompanyName": customer.company_name,
            "PrimaryEmailAddr": {
                "Address": customer.email
            },
            "PrimaryPhone": {
                "FreeFormNumber": customer.phone
            },
            "BillAddr": {
                "Line1": customer.billing_address.line1 if customer.billing_address else None,
                "City": customer.billing_address.city if customer.billing_address else None,
                "CountrySubDivisionCode": customer.billing_address.state if customer.billing_address else None,
                "PostalCode": customer.billing_address.postal_code if customer.billing_address else None,
                "Country": customer.billing_address.country if customer.billing_address else None
            }
        }
    
    async def update_customer(self, customer: Customer) -> Dict[str, Any]:
        """Update an existing customer."""
        # In a real application, this would call the QuickBooks API
        # For documentation purposes, we'll simulate the response
        
        return {
            "Id": customer.external_id,
            "SyncToken": "1",
            "DisplayName": customer.display_name,
            "GivenName": customer.first_name,
            "FamilyName": customer.last_name,
            "CompanyName": customer.company_name,
            "PrimaryEmailAddr": {
                "Address": customer.email
            },
            "PrimaryPhone": {
                "FreeFormNumber": customer.phone
            },
            "BillAddr": {
                "Line1": customer.billing_address.line1 if customer.billing_address else None,
                "City": customer.billing_address.city if customer.billing_address else None,
                "CountrySubDivisionCode": customer.billing_address.state if customer.billing_address else None,
                "PostalCode": customer.billing_address.postal_code if customer.billing_address else None,
                "Country": customer.billing_address.country if customer.billing_address else None
            }
        }
    
    async def get_invoice(self, invoice_id: str) -> Optional[Dict[str, Any]]:
        """Get an invoice by ID."""
        # In a real application, this would query the QuickBooks API
        # For documentation purposes, we'll simulate the response
        
        return {
            "Id": invoice_id,
            "DocNumber": "INV-001",
            "CustomerRef": {
                "value": "123456",
                "name": "ACME Corporation"
            },
            "TxnDate": "2023-04-01",
            "DueDate": "2023-05-01",
            "TotalAmt": 1500.00,
            "Balance": 1500.00,
            "Line": [
                {
                    "DetailType": "SalesItemLineDetail",
                    "Amount": 1000.00,
                    "Description": "Consulting Services",
                    "SalesItemLineDetail": {
                        "ItemRef": {
                            "value": "1",
                            "name": "Consulting"
                        },
                        "Qty": 10,
                        "UnitPrice": 100.00
                    }
                },
                {
                    "DetailType": "SalesItemLineDetail",
                    "Amount": 500.00,
                    "Description": "Software License",
                    "SalesItemLineDetail": {
                        "ItemRef": {
                            "value": "2",
                            "name": "Software License"
                        },
                        "Qty": 1,
                        "UnitPrice": 500.00
                    }
                }
            ]
        }
    
    async def create_invoice(self, invoice: Invoice) -> Dict[str, Any]:
        """Create a new invoice."""
        # In a real application, this would call the QuickBooks API
        # For documentation purposes, we'll simulate the response
        
        return {
            "Id": "987654321",
            "SyncToken": "0",
            "DocNumber": invoice.doc_number or "INV-001",
            "CustomerRef": {
                "value": invoice.customer_id,
                "name": invoice.customer_name
            },
            "TxnDate": invoice.date.isoformat(),
            "DueDate": invoice.due_date.isoformat() if invoice.due_date else None,
            "TotalAmt": invoice.total.value,
            "Balance": invoice.balance.value
        }


class QuickBooksContextHandler(ContextHandler):
    """Handler for processing MCP contexts for QuickBooks."""
    
    def __init__(self, service: QuickBooksService):
        """Initialize QuickBooks context handler."""
        self.service = service
        self.handlers = {
            "quickbooks.company.get": self._handle_company_get,
            "quickbooks.customer.get": self._handle_customer_get,
            "quickbooks.customer.create": self._handle_customer_create,
            "quickbooks.customer.update": self._handle_customer_update,
            "quickbooks.invoice.get": self._handle_invoice_get,
            "quickbooks.invoice.create": self._handle_invoice_create
        }
    
    async def can_handle(self, context: Context) -> bool:
        """Check if this handler can process the given context."""
        return context.name.startswith("quickbooks.")
    
    async def handle(self, context: Context) -> ContextResult:
        """Handle the given context."""
        handler = self.handlers.get(context.name)
        
        if not handler:
            return ContextResult(
                success=False,
                error=f"Unsupported context: {context.name}",
                data=None
            )
            
        try:
            return await handler(context)
        except Exception as e:
            logger.error(f"Error handling context {context.name}: {str(e)}")
            return ContextResult(
                success=False,
                error=str(e),
                data=None
            )
    
    async def _handle_company_get(self, context: Context) -> ContextResult:
        """Handle quickbooks.company.get context."""
        company_info = await self.service.get_company_info()
        
        return ContextResult(
            success=True,
            error=None,
            data=company_info
        )
    
    async def _handle_customer_get(self, context: Context) -> ContextResult:
        """Handle quickbooks.customer.get context."""
        customer_id = context.parameters.get("customer_id")
        if not customer_id:
            return ContextResult(
                success=False,
                error="customer_id is required",
                data=None
            )
            
        customer = await self.service.get_customer(customer_id)
        
        if not customer:
            return ContextResult(
                success=False,
                error=f"Customer not found: {customer_id}",
                data=None
            )
            
        return ContextResult(
            success=True,
            error=None,
            data=customer
        )
    
    async def _handle_customer_create(self, context: Context) -> ContextResult:
        """Handle quickbooks.customer.create context."""
        customer_data = context.parameters
        
        # Create customer entity
        customer = Customer(
            display_name=customer_data.get("display_name", ""),
            first_name=customer_data.get("first_name"),
            last_name=customer_data.get("last_name"),
            company_name=customer_data.get("company_name"),
            email=customer_data.get("email"),
            phone=customer_data.get("phone")
        )
        
        # Add billing address if provided
        if "billing_address" in customer_data:
            billing_data = customer_data["billing_address"]
            customer.billing_address = Address(
                line1=billing_data.get("line1"),
                line2=billing_data.get("line2"),
                city=billing_data.get("city"),
                state=billing_data.get("state"),
                postal_code=billing_data.get("postal_code"),
                country=billing_data.get("country")
            )
        
        # Validate customer
        validation_errors = customer.validate()
        if validation_errors:
            return ContextResult(
                success=False,
                error=f"Customer validation failed: {', '.join(validation_errors)}",
                data=None
            )
            
        # Create customer in QuickBooks
        qbo_customer = await self.service.create_customer(customer)
        
        return ContextResult(
            success=True,
            error=None,
            data=qbo_customer
        )
    
    async def _handle_customer_update(self, context: Context) -> ContextResult:
        """Handle quickbooks.customer.update context."""
        customer_data = context.parameters
        
        if not customer_data.get("id") and not customer_data.get("external_id"):
            return ContextResult(
                success=False,
                error="id or external_id is required",
                data=None
            )
            
        # Create customer entity
        customer = Customer(
            id=customer_data.get("id", ""),
            external_id=customer_data.get("external_id"),
            display_name=customer_data.get("display_name", ""),
            first_name=customer_data.get("first_name"),
            last_name=customer_data.get("last_name"),
            company_name=customer_data.get("company_name"),
            email=customer_data.get("email"),
            phone=customer_data.get("phone")
        )
        
        # Add billing address if provided
        if "billing_address" in customer_data:
            billing_data = customer_data["billing_address"]
            customer.billing_address = Address(
                line1=billing_data.get("line1"),
                line2=billing_data.get("line2"),
                city=billing_data.get("city"),
                state=billing_data.get("state"),
                postal_code=billing_data.get("postal_code"),
                country=billing_data.get("country")
            )
        
        # Update customer in QuickBooks
        qbo_customer = await self.service.update_customer(customer)
        
        return ContextResult(
            success=True,
            error=None,
            data=qbo_customer
        )
    
    async def _handle_invoice_get(self, context: Context) -> ContextResult:
        """Handle quickbooks.invoice.get context."""
        invoice_id = context.parameters.get("invoice_id")
        if not invoice_id:
            return ContextResult(
                success=False,
                error="invoice_id is required",
                data=None
            )
            
        invoice = await self.service.get_invoice(invoice_id)
        
        if not invoice:
            return ContextResult(
                success=False,
                error=f"Invoice not found: {invoice_id}",
                data=None
            )
            
        return ContextResult(
            success=True,
            error=None,
            data=invoice
        )
    
    async def _handle_invoice_create(self, context: Context) -> ContextResult:
        """Handle quickbooks.invoice.create context."""
        invoice_data = context.parameters
        
        if not invoice_data.get("customer_id"):
            return ContextResult(
                success=False,
                error="customer_id is required",
                data=None
            )
            
        if not invoice_data.get("items") or not isinstance(invoice_data.get("items"), list):
            return ContextResult(
                success=False,
                error="items is required and must be a list",
                data=None
            )
            
        # Create invoice items
        items = []
        for item_data in invoice_data.get("items", []):
            item = InvoiceItem(
                description=item_data.get("description", ""),
                quantity=float(item_data.get("quantity", 1)),
                unit_price=Money(float(item_data.get("unit_price", 0))),
                item_id=item_data.get("item_id")
            )
            items.append(item)
        
        # Create invoice entity
        invoice = Invoice(
            customer_id=invoice_data.get("customer_id"),
            customer_name=invoice_data.get("customer_name"),
            doc_number=invoice_data.get("doc_number"),
            line_items=items,
            notes=invoice_data.get("notes"),
            terms=invoice_data.get("terms")
        )
        
        # Validate invoice
        validation_errors = invoice.validate()
        if validation_errors:
            return ContextResult(
                success=False,
                error=f"Invoice validation failed: {', '.join(validation_errors)}",
                data=None
            )
            
        # Create invoice in QuickBooks
        qbo_invoice = await self.service.create_invoice(invoice)
        
        return ContextResult(
            success=True,
            error=None,
            data=qbo_invoice
        )


# Example usage
async def example_usage():
    """Demonstrate usage of the QuickBooks integration."""
    # Create configuration
    config = QboConfig(
        client_id="your_client_id",
        client_secret="your_client_secret",
        redirect_uri="https://your-app.example.com/callback",
        environment=QboEnvironment.SANDBOX
    )
    
    # Create auth manager
    auth_manager = QboAuthManager(config)
    
    # Create service
    service = QuickBooksService(config, auth_manager)
    
    # Create context handler
    context_handler = QuickBooksContextHandler(service)
    
    # Get authorization URL
    auth_url = auth_manager.get_authorization_url()
    print(f"Please visit the following URL to authorize the application: {auth_url}")
    
    # In a real application, the user would visit this URL and authenticate
    # After authentication, they would be redirected back to your application
    # with an authorization code and realm ID in the URL
    
    # For demonstration, we'll simulate this
    auth_code = "simulated_auth_code"
    realm_id = "123456789"
    
    # Exchange authorization code for access token
    token_data = await auth_manager.exchange_code_for_token(auth_code, realm_id)
    print(f"Received token data: {token_data}")
    
    # Get company info
    company_info = await service.get_company_info()
    print(f"Company info: {company_info}")
    
    # Get a customer
    customer = await service.get_customer("123456")
    print(f"Customer: {customer}")
    
    # Create a new customer
    new_customer = Customer(
        display_name="New Customer",
        first_name="John",
        last_name="Smith",
        company_name="New Company",
        email="john.smith@example.com",
        phone="555-987-6543",
        billing_address=Address(
            line1="456 Market St",
            city="San Francisco",
            state="CA",
            postal_code="94105",
            country="US"
        )
    )
    created_customer = await service.create_customer(new_customer)
    print(f"Created customer: {created_customer}")
    
    # Create invoice items
    items = [
        InvoiceItem(
            description="Consulting Services",
            quantity=10,
            unit_price=Money(150.0)
        ),
        InvoiceItem(
            description="Software License",
            quantity=1,
            unit_price=Money(499.99)
        )
    ]
    
    # Create an invoice
    invoice = Invoice(
        customer_id="123456",
        customer_name="ACME Corporation",
        line_items=items,
        terms="Net 30",
        notes="Thank you for your business!"
    )
    
    created_invoice = await service.create_invoice(invoice)
    print(f"Created invoice: {created_invoice}")


if __name__ == "__main__":
    asyncio.run(example_usage()) 