#!/usr/bin/env python3
"""
DocuChat Web Interface

Web-based interface for DocuChat v2.0 using FastAPI and Streamlit.
Provides REST API endpoints and interactive web UI.
"""

__version__ = "2.0.0"

try:
    from .api import app as fastapi_app
except ImportError:
    # Web dependencies not installed
    fastapi_app = None

try:
    from .streamlit_app import main as streamlit_main
except ImportError:
    # Streamlit not installed
    streamlit_main = None

__all__ = ['fastapi_app', 'streamlit_main']
