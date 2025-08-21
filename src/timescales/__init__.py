from importlib import metadata as _metadata
try:
    __version__ = _metadata.version("timescales")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0+src"


from .api import TimescaleEnsemble

# from .physics.timescales import relaxation_timescale, collision_timescale

from .profiles.power_law import PowerLawProfile

from .factory import create_profile, available_profiles

from . import profiles  # noqa: F401
from . import physics   # noqa: F401

__all__ = [
    # Version
    "__version__",
    # Orchestrator
    "TimescaleEnsemble",
    # Physics (model-agnostic)
    "relaxation_timescale",
    "collision_timescale",
    # Profiles
    "PowerLawProfile",
    # Factory helpers
    "create_profile",
    "available_profiles",
    # (Optional) subpackages
    "profiles",
    "physics",
]
