"""Invoice domain tool definitions."""

from __future__ import annotations

from typing import List

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool, tool


def create_invoice_tools(db: SQLDatabase) -> List[BaseTool]:
    """Create all invoice-related tools bound to the provided database."""

    @tool
    def get_invoices_by_customer_sorted_by_date(customer_id: str) -> str:
        """Look up invoices for a customer sorted by invoice date (descending)."""
        return db.run(
            "SELECT * FROM Invoice "
            f"WHERE CustomerId = {customer_id} "
            "ORDER BY InvoiceDate DESC;"
        )

    @tool
    def get_invoices_sorted_by_unit_price(customer_id: str) -> str:
        """Look up customer invoices sorted by highest line-item unit price."""
        return db.run(
            f"""
            SELECT Invoice.*, InvoiceLine.UnitPrice
            FROM Invoice
            JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId
            WHERE Invoice.CustomerId = {customer_id}
            ORDER BY InvoiceLine.UnitPrice DESC;
            """
        )

    @tool
    def get_employee_by_invoice_and_customer(invoice_id: str, customer_id: str) -> str:
        """Get support employee info associated with an invoice and customer."""
        return db.run(
            f"""
            SELECT Employee.FirstName, Employee.Title, Employee.Email
            FROM Employee
            JOIN Customer ON Customer.SupportRepId = Employee.EmployeeId
            JOIN Invoice ON Invoice.CustomerId = Customer.CustomerId
            WHERE Invoice.InvoiceId = ({invoice_id}) AND Invoice.CustomerId = ({customer_id});
            """,
            include_columns=True,
        )

    return [
        get_invoices_by_customer_sorted_by_date,
        get_invoices_sorted_by_unit_price,
        get_employee_by_invoice_and_customer,
    ]
