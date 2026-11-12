import os

# Reduce memory/CPU thread usage on Render's free tier before importing heavy libs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys

# Add the backend/ directory to Python path so all sibling imports
# inside backend/ resolve correctly (rag_system, config, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

# Import the Flask app — now resolves as plain `app` module in path
from app import app  # noqa: E402

# Expose for gunicorn: gunicorn wsgi:app
__all__ = ['app']
