#!/usr/bin/env python3
"""
MCP Accounting Entity Models
===========================

This module provides data models for accounting entities used in the integration
with accounting software like QuickBooks. These models provide a standardized 
interface that abstracts away the specifics of various accounting platforms.

Key components:
- Base Entity model with common functionality
- Customer, Invoice, Payment, and Item models
- Serialization and validation logic
"""

import uuid
import json
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, ClassVar
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod


# ===== Base Types =====

class EntityType(Enum):
    """Types of accounting entities."""
    CUSTOMER = "customer"
    INVOICE = "invoice"
    PAYMENT = "payment"
    ITEM = "item"
    VENDOR = "vendor"
    BILL = "bill"
    ACCOUNT = "account"
    TRANSACTION = "transaction"


class EntityStatus(Enum):
    """Common entity status values."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class Money:
    """Represents a monetary value with currency."""
    value: float
    currency: str = "USD"
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.value} {self.currency}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "currency": self.currency
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Money':
        """Create from dictionary."""
        return cls(
            value=float(data.get("value", 0.0)),
            currency=data.get("currency", "USD")
        )


@dataclass
class Address:
    """Represents a physical address."""
    line1: Optional[str] = None
    line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = "US"
    
    def is_valid(self) -> bool:
        """Check if address has minimal required fields."""
        return bool(self.line1 and self.city and self.postal_code)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Address':
        """Create from dictionary."""
        return cls(
            line1=data.get("line1"),
            line2=data.get("line2"),
            city=data.get("city"),
            state=data.get("state"),
            postal_code=data.get("postal_code"),
            country=data.get("country", "US")
        )


# ===== Base Entity =====

@dataclass
class Entity(ABC):
    """Base class for all accounting entities."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    external_id: Optional[str] = None
    entity_type: ClassVar[EntityType]
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    source_system: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        result = {
            "id": self.id,
            "type": self.entity_type.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if self.external_id:
            result["external_id"] = self.external_id
            
        if self.source_system:
            result["source_system"] = self.source_system
            
        return result
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate entity and return list of validation errors."""
        pass
    
    def is_valid(self) -> bool:
        """Check if entity is valid."""
        return len(self.validate()) == 0


# ===== Customer Entity =====

@dataclass
class Customer(Entity):
    """Represents a customer in an accounting system."""
    entity_type: ClassVar[EntityType] = EntityType.CUSTOMER
    display_name: str = ""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    status: EntityStatus = EntityStatus.ACTIVE
    billing_address: Optional[Address] = None
    shipping_address: Optional[Address] = None
    tax_identifier: Optional[str] = None
    notes: Optional[str] = None
    balance: Money = field(default_factory=lambda: Money(0.0))
    currency: str = "USD"
    
    def validate(self) -> List[str]:
        """Validate customer and return list of validation errors."""
        errors = []
        
        if not self.display_name:
            errors.append("display_name is required")
            
        if not (self.first_name or self.last_name or self.company_name):
            errors.append("At least one of first_name, last_name, or company_name is required")
            
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert customer to dictionary."""
        result = super().to_dict()
        
        result.update({
            "display_name": self.display_name,
            "status": self.status.value,
            "balance": self.balance.to_dict(),
            "currency": self.currency
        })
        
        if self.first_name:
            result["first_name"] = self.first_name
            
        if self.last_name:
            result["last_name"] = self.last_name
            
        if self.company_name:
            result["company_name"] = self.company_name
            
        if self.email:
            result["email"] = self.email
            
        if self.phone:
            result["phone"] = self.phone
            
        if self.billing_address:
            result["billing_address"] = self.billing_address.to_dict()
            
        if self.shipping_address:
            result["shipping_address"] = self.shipping_address.to_dict()
            
        if self.tax_identifier:
            result["tax_identifier"] = self.tax_identifier
            
        if self.notes:
            result["notes"] = self.notes
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Customer':
        """Create from dictionary."""
        customer = cls(
            id=data.get("id", str(uuid.uuid4())),
            external_id=data.get("external_id"),
            display_name=data.get("display_name", ""),
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            company_name=data.get("company_name"),
            email=data.get("email"),
            phone=data.get("phone"),
            status=EntityStatus(data.get("status", "active")),
            tax_identifier=data.get("tax_identifier"),
            notes=data.get("notes"),
            currency=data.get("currency", "USD"),
            source_system=data.get("source_system")
        )
        
        if "created_at" in data:
            try:
                customer.created_at = datetime.datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                pass
                
        if "updated_at" in data:
            try:
                customer.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
            except (ValueError, TypeError):
                pass
                
        if "balance" in data:
            customer.balance = Money.from_dict(data["balance"]) if isinstance(data["balance"], dict) else Money(float(data["balance"]))
            
        if "billing_address" in data and data["billing_address"]:
            customer.billing_address = Address.from_dict(data["billing_address"])
            
        if "shipping_address" in data and data["shipping_address"]:
            customer.shipping_address = Address.from_dict(data["shipping_address"])
            
        return customer


# ===== Invoice Item =====

@dataclass
class InvoiceItem:
    """Represents an item line in an invoice."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    quantity: float = 1.0
    unit_price: Money = field(default_factory=lambda: Money(0.0))
    amount: Money = field(default_factory=lambda: Money(0.0))
    item_id: Optional[str] = None
    item_name: Optional[str] = None
    tax_rate: Optional[float] = None
    tax_amount: Optional[Money] = None
    discount_rate: Optional[float] = None
    discount_amount: Optional[Money] = None
    
    def __post_init__(self):
        """Calculate amount if not provided."""
        if self.amount.value == 0.0 and self.unit_price.value != 0.0:
            self.amount = Money(
                value=round(self.quantity * self.unit_price.value, 2),
                currency=self.unit_price.currency
            )
    
    def validate(self) -> List[str]:
        """Validate invoice item and return list of validation errors."""
        errors = []
        
        if not self.description:
            errors.append("description is required")
            
        if self.quantity <= 0:
            errors.append("quantity must be greater than 0")
            
        if self.unit_price.value < 0:
            errors.append("unit_price must be greater than or equal to 0")
            
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert invoice item to dictionary."""
        result = {
            "id": self.id,
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": self.unit_price.to_dict(),
            "amount": self.amount.to_dict()
        }
        
        if self.item_id:
            result["item_id"] = self.item_id
            
        if self.item_name:
            result["item_name"] = self.item_name
            
        if self.tax_rate is not None:
            result["tax_rate"] = self.tax_rate
            
        if self.tax_amount:
            result["tax_amount"] = self.tax_amount.to_dict()
            
        if self.discount_rate is not None:
            result["discount_rate"] = self.discount_rate
            
        if self.discount_amount:
            result["discount_amount"] = self.discount_amount.to_dict()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InvoiceItem':
        """Create from dictionary."""
        item = cls(
            id=data.get("id", str(uuid.uuid4())),
            description=data.get("description", ""),
            quantity=float(data.get("quantity", 1.0)),
            item_id=data.get("item_id"),
            item_name=data.get("item_name")
        )
        
        if "unit_price" in data:
            item.unit_price = Money.from_dict(data["unit_price"]) if isinstance(data["unit_price"], dict) else Money(float(data["unit_price"]))
            
        if "amount" in data:
            item.amount = Money.from_dict(data["amount"]) if isinstance(data["amount"], dict) else Money(float(data["amount"]))
        
        if "tax_rate" in data:
            item.tax_rate = float(data["tax_rate"])
            
        if "tax_amount" in data:
            item.tax_amount = Money.from_dict(data["tax_amount"]) if isinstance(data["tax_amount"], dict) else Money(float(data["tax_amount"]))
            
        if "discount_rate" in data:
            item.discount_rate = float(data["discount_rate"])
            
        if "discount_amount" in data:
            item.discount_amount = Money.from_dict(data["discount_amount"]) if isinstance(data["discount_amount"], dict) else Money(float(data["discount_amount"]))
            
        return item


# ===== Invoice Entity =====

class InvoiceStatus(Enum):
    """Possible invoice status values."""
    DRAFT = "draft"
    SENT = "sent"
    VIEWED = "viewed"
    PAID = "paid"
    PARTIALLY_PAID = "partially_paid"
    OVERDUE = "overdue"
    VOID = "void"
    CANCELLED = "cancelled"


@dataclass
class Invoice(Entity):
    """Represents an invoice in an accounting system."""
    entity_type: ClassVar[EntityType] = EntityType.INVOICE
    doc_number: Optional[str] = None
    customer_id: str = ""
    customer_name: Optional[str] = None
    date: datetime.date = field(default_factory=datetime.date.today)
    due_date: Optional[datetime.date] = None
    status: InvoiceStatus = InvoiceStatus.DRAFT
    currency: str = "USD"
    line_items: List[InvoiceItem] = field(default_factory=list)
    subtotal: Money = field(default_factory=lambda: Money(0.0))
    tax_total: Money = field(default_factory=lambda: Money(0.0))
    discount_total: Money = field(default_factory=lambda: Money(0.0))
    total: Money = field(default_factory=lambda: Money(0.0))
    balance: Money = field(default_factory=lambda: Money(0.0))
    notes: Optional[str] = None
    terms: Optional[str] = None
    
    def __post_init__(self):
        """Calculate totals if not provided."""
        self._recalculate_totals()
    
    def _recalculate_totals(self):
        """Recalculate invoice totals based on line items."""
        if not self.line_items:
            return
            
        currency = self.line_items[0].amount.currency
        
        # Calculate subtotal
        subtotal = sum(item.amount.value for item in self.line_items)
        self.subtotal = Money(value=round(subtotal, 2), currency=currency)
        
        # Calculate tax total
        tax_total = sum(
            (item.tax_amount.value if item.tax_amount else 
             (item.amount.value * item.tax_rate / 100 if item.tax_rate else 0))
            for item in self.line_items
        )
        self.tax_total = Money(value=round(tax_total, 2), currency=currency)
        
        # Calculate discount total
        discount_total = sum(
            (item.discount_amount.value if item.discount_amount else 
             (item.amount.value * item.discount_rate / 100 if item.discount_rate else 0))
            for item in self.line_items
        )
        self.discount_total = Money(value=round(discount_total, 2), currency=currency)
        
        # Calculate total
        total = subtotal + tax_total - discount_total
        self.total = Money(value=round(total, 2), currency=currency)
        
        # If balance not explicitly set, default to total
        if self.balance.value == 0.0:
            self.balance = Money(value=self.total.value, currency=currency)
    
    def validate(self) -> List[str]:
        """Validate invoice and return list of validation errors."""
        errors = []
        
        if not self.customer_id:
            errors.append("customer_id is required")
            
        if not self.line_items:
            errors.append("At least one line item is required")
        else:
            for i, item in enumerate(self.line_items):
                item_errors = item.validate()
                for error in item_errors:
                    errors.append(f"Line item {i+1}: {error}")
                    
        if self.due_date and self.due_date < self.date:
            errors.append("due_date cannot be earlier than invoice date")
            
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert invoice to dictionary."""
        result = super().to_dict()
        
        result.update({
            "customer_id": self.customer_id,
            "date": self.date.isoformat() if self.date else None,
            "status": self.status.value,
            "currency": self.currency,
            "line_items": [item.to_dict() for item in self.line_items],
            "subtotal": self.subtotal.to_dict(),
            "tax_total": self.tax_total.to_dict(),
            "discount_total": self.discount_total.to_dict(),
            "total": self.total.to_dict(),
            "balance": self.balance.to_dict()
        })
        
        if self.doc_number:
            result["doc_number"] = self.doc_number
            
        if self.customer_name:
            result["customer_name"] = self.customer_name
            
        if self.due_date:
            result["due_date"] = self.due_date.isoformat()
            
        if self.notes:
            result["notes"] = self.notes
            
        if self.terms:
            result["terms"] = self.terms
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Invoice':
        """Create from dictionary."""
        invoice = cls(
            id=data.get("id", str(uuid.uuid4())),
            external_id=data.get("external_id"),
            doc_number=data.get("doc_number"),
            customer_id=data.get("customer_id", ""),
            customer_name=data.get("customer_name"),
            status=InvoiceStatus(data.get("status", "draft")),
            currency=data.get("currency", "USD"),
            notes=data.get("notes"),
            terms=data.get("terms"),
            source_system=data.get("source_system")
        )
        
        # Parse dates
        if "date" in data:
            try:
                invoice.date = datetime.date.fromisoformat(data["date"]) if isinstance(data["date"], str) else data["date"]
            except (ValueError, TypeError):
                pass
                
        if "due_date" in data and data["due_date"]:
            try:
                invoice.due_date = datetime.date.fromisoformat(data["due_date"]) if isinstance(data["due_date"], str) else data["due_date"]
            except (ValueError, TypeError):
                pass
        
        # Parse timestamps
        if "created_at" in data:
            try:
                invoice.created_at = datetime.datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                pass
                
        if "updated_at" in data:
            try:
                invoice.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
            except (ValueError, TypeError):
                pass
        
        # Parse money fields
        if "subtotal" in data:
            invoice.subtotal = Money.from_dict(data["subtotal"]) if isinstance(data["subtotal"], dict) else Money(float(data["subtotal"]))
            
        if "tax_total" in data:
            invoice.tax_total = Money.from_dict(data["tax_total"]) if isinstance(data["tax_total"], dict) else Money(float(data["tax_total"]))
            
        if "discount_total" in data:
            invoice.discount_total = Money.from_dict(data["discount_total"]) if isinstance(data["discount_total"], dict) else Money(float(data["discount_total"]))
            
        if "total" in data:
            invoice.total = Money.from_dict(data["total"]) if isinstance(data["total"], dict) else Money(float(data["total"]))
            
        if "balance" in data:
            invoice.balance = Money.from_dict(data["balance"]) if isinstance(data["balance"], dict) else Money(float(data["balance"]))
        
        # Parse line items
        if "line_items" in data and isinstance(data["line_items"], list):
            invoice.line_items = [InvoiceItem.from_dict(item) for item in data["line_items"]]
            
        return invoice


# ===== Payment Entity =====

class PaymentMethod(Enum):
    """Possible payment method types."""
    CASH = "cash"
    CHECK = "check"
    CREDIT_CARD = "credit_card"
    ACH = "ach"
    WIRE_TRANSFER = "wire_transfer"
    PAYPAL = "paypal"
    OTHER = "other"


@dataclass
class Payment(Entity):
    """Represents a payment in an accounting system."""
    entity_type: ClassVar[EntityType] = EntityType.PAYMENT
    customer_id: str = ""
    customer_name: Optional[str] = None
    date: datetime.date = field(default_factory=datetime.date.today)
    amount: Money = field(default_factory=lambda: Money(0.0))
    currency: str = "USD"
    payment_method: PaymentMethod = PaymentMethod.OTHER
    reference: Optional[str] = None
    memo: Optional[str] = None
    applied_to: List[Dict[str, Any]] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate payment and return list of validation errors."""
        errors = []
        
        if not self.customer_id:
            errors.append("customer_id is required")
            
        if self.amount.value <= 0:
            errors.append("amount must be greater than 0")
            
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert payment to dictionary."""
        result = super().to_dict()
        
        result.update({
            "customer_id": self.customer_id,
            "date": self.date.isoformat() if self.date else None,
            "amount": self.amount.to_dict(),
            "currency": self.currency,
            "payment_method": self.payment_method.value,
            "applied_to": self.applied_to
        })
        
        if self.customer_name:
            result["customer_name"] = self.customer_name
            
        if self.reference:
            result["reference"] = self.reference
            
        if self.memo:
            result["memo"] = self.memo
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Payment':
        """Create from dictionary."""
        payment = cls(
            id=data.get("id", str(uuid.uuid4())),
            external_id=data.get("external_id"),
            customer_id=data.get("customer_id", ""),
            customer_name=data.get("customer_name"),
            payment_method=PaymentMethod(data.get("payment_method", "other")),
            reference=data.get("reference"),
            memo=data.get("memo"),
            currency=data.get("currency", "USD"),
            source_system=data.get("source_system")
        )
        
        # Parse date
        if "date" in data:
            try:
                payment.date = datetime.date.fromisoformat(data["date"]) if isinstance(data["date"], str) else data["date"]
            except (ValueError, TypeError):
                pass
        
        # Parse timestamps
        if "created_at" in data:
            try:
                payment.created_at = datetime.datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                pass
                
        if "updated_at" in data:
            try:
                payment.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
            except (ValueError, TypeError):
                pass
        
        # Parse amount
        if "amount" in data:
            payment.amount = Money.from_dict(data["amount"]) if isinstance(data["amount"], dict) else Money(float(data["amount"]))
        
        # Parse applied_to
        if "applied_to" in data and isinstance(data["applied_to"], list):
            payment.applied_to = data["applied_to"]
            
        return payment


# ===== Usage Examples =====

def example_usage():
    """Demonstrate usage of the entity models."""
    # Create a customer
    customer = Customer(
        display_name="Acme Corp",
        company_name="Acme Corporation",
        first_name="John",
        last_name="Doe",
        email="john.doe@acme.com",
        phone="555-123-4567",
        billing_address=Address(
            line1="123 Main St",
            city="Anytown",
            state="CA",
            postal_code="12345"
        )
    )
    
    # Validate customer
    validation_errors = customer.validate()
    print(f"Customer validation errors: {validation_errors}")
    
    # Create invoice items
    items = [
        InvoiceItem(
            description="Consulting Services",
            quantity=10,
            unit_price=Money(150.0),
            tax_rate=8.25
        ),
        InvoiceItem(
            description="Software License",
            quantity=1,
            unit_price=Money(499.99),
            tax_rate=8.25
        )
    ]
    
    # Create an invoice
    invoice = Invoice(
        customer_id=customer.id,
        customer_name=customer.display_name,
        doc_number="INV-001",
        line_items=items,
        terms="Net 30",
        notes="Thank you for your business!"
    )
    
    # Validate invoice
    validation_errors = invoice.validate()
    print(f"Invoice validation errors: {validation_errors}")
    
    # Display invoice totals
    print(f"Invoice subtotal: {invoice.subtotal}")
    print(f"Invoice tax: {invoice.tax_total}")
    print(f"Invoice total: {invoice.total}")
    
    # Create a payment
    payment = Payment(
        customer_id=customer.id,
        customer_name=customer.display_name,
        amount=Money(invoice.total.value),
        payment_method=PaymentMethod.CREDIT_CARD,
        reference="CARD-1234",
        applied_to=[
            {
                "invoice_id": invoice.id,
                "amount": invoice.total.value
            }
        ]
    )
    
    # Validate payment
    validation_errors = payment.validate()
    print(f"Payment validation errors: {validation_errors}")
    
    # Convert to/from JSON
    customer_json = json.dumps(customer.to_dict(), indent=2)
    print(f"Customer JSON: {customer_json}")
    
    # Recreate from JSON
    customer_dict = json.loads(customer_json)
    recreated_customer = Customer.from_dict(customer_dict)
    print(f"Recreated customer: {recreated_customer.display_name}")


if __name__ == "__main__":
    example_usage() 