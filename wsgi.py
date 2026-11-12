import os

# Reduce memory/CPU thread usage on Render's free tier before importing heavy libs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class LazyApplication:
    """Defer importing the heavy Flask app until the first request.

    Gunicorn imports ``wsgi:app`` in the master process before binding the port.
    Importing the real app pulls in fastembed/torch/numpy/pandas/faiss, which can
    delay or prevent port binding on Render. This wrapper keeps the import out of
    the master process so the port binds immediately.
    """

    def __init__(self):
        self._app = None

    def _load(self):
        if self._app is None:
            import sys

            backend_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "backend"
            )
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)
            from app import app as real_app

            self._app = real_app
        return self._app

    def __call__(self, environ, start_response):
        return self._load()(environ, start_response)


app = LazyApplication()

# Expose for gunicorn: gunicorn wsgi:app
__all__ = ["app"]