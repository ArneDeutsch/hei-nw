"""HEI NW package."""

from pkgutil import extend_path

from .gate import NeuromodulatedGate, SalienceFeatures
from .keyer import DGKeyer
from .recall import RecallService
from .store import EpisodicStore

__path__ = extend_path(__path__, __name__)

__all__ = [
    "__version__",
    "DGKeyer",
    "EpisodicStore",
    "NeuromodulatedGate",
    "RecallService",
    "SalienceFeatures",
]

__version__ = "0.0.0"
