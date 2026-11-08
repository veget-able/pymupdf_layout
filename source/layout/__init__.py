from . import DocumentLayoutAnalyzer
from . import onnx
from . import pymupdf_util

from pymupdf.layout._build import git_sha
from pymupdf.layout._build import platform_python_implementation
from pymupdf.layout._build import version
from pymupdf.layout._build import version_tuple

import pymupdf


_ACTIVE_MODEL = None
_ACTIVE_PREDICTOR = None
_REGISTERED_PROVIDER = None


def get_active_provider():
    if _REGISTERED_PROVIDER is not None and pymupdf._get_layout is not _REGISTERED_PROVIDER[1]:
        raise RuntimeError("Registered layout provider hook changed within scope")
    return _REGISTERED_PROVIDER[0] if _REGISTERED_PROVIDER is not None else None


from contextlib import contextmanager
from threading import RLock
_PROVIDER_LOCK = RLock()


@contextmanager
def use_provider(provider):
    """Own the explicit custom provider and restore the previous global hook."""
    global _REGISTERED_PROVIDER
    with _PROVIDER_LOCK:
        previous = (pymupdf._get_layout, _REGISTERED_PROVIDER)
        hook = provider.predict
        _REGISTERED_PROVIDER = (provider, hook)
        pymupdf._get_layout = hook
        try:
            yield provider
        finally:
            pymupdf._get_layout, _REGISTERED_PROVIDER = previous



def get_active_model():
    """Borrow the already-active native model without constructing a new one.

    The provider owns this model. Consumers may build local views sharing its
    sessions, but must not mutate the shared model, extractor or session.
    Custom layout providers must expose their own explicit model contract.
    """
    provider = get_active_provider()
    if provider is not None:
        return provider.model
    activate()
    if _ACTIVE_MODEL is None or pymupdf._get_layout is not _ACTIVE_PREDICTOR:
        raise RuntimeError("The active layout provider has no registered native model")
    return _ACTIVE_MODEL


def activate():
    """Create a layout analyzer function using an ONNX model."""
    if callable(pymupdf._get_layout):
        return  # already activated
    global _ACTIVE_MODEL, _ACTIVE_PREDICTOR
    MODEL = DocumentLayoutAnalyzer.get_model()

    def _get_layout(*args, **kwargs):
        return MODEL.predict(*args, **kwargs)

    _ACTIVE_MODEL = MODEL
    _ACTIVE_PREDICTOR = _get_layout
    pymupdf._get_layout = _get_layout


activate()
