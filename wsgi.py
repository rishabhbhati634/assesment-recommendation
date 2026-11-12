import sys
import os

# Add the backend/ directory to Python path so all sibling imports
# inside backend/ resolve correctly (rag_system, config, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

# Import the Flask app — now resolves as plain `app` module in path
from app import app  # noqa: E402

# Expose for gunicorn: gunicorn wsgi:app
__all__ = ['app']
