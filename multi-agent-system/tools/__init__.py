"""Tool factories for music and invoice domains."""

from tools.invoice import create_invoice_tools
from tools.music import create_music_tools

__all__ = ["create_invoice_tools", "create_music_tools"]
