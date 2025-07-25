#!/usr/bin/env python3
"""
DocuChat CLI Interface

Command-line interface for DocuChat v2.0 with interactive terminal,
real-time file monitoring, and comprehensive document chat capabilities.
"""

__version__ = "2.0.0"

from .main import DocuChatApp, main

__all__ = ['DocuChatApp', 'main']
