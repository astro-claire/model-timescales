# src/timescales/factory.py
"""
Profile factory & registry.

- Profiles register themselves via @register_profile("powerlaw", aliases=("pl",))
- Users (and the API layer) ask for profiles by name via create_profile("powerlaw", **cfg)
- available_profiles() lists what's registered.

Optional: plugin discovery via entry points (group: "timescales.profiles").
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Callable, Dict, Iterable, List, Optional, Type, TypeVar

from .profiles.base import ProfileBase  # your abstract interface

T = TypeVar("T", bound=ProfileBase)

# -----------------------------------------------------------------------------
# Internal registries
# -----------------------------------------------------------------------------
_REGISTRY: Dict[str, Type[ProfileBase]] = {}   # canonical_name -> class
_ALIASES: Dict[str, str] = {}                  # alias_name -> canonical_name
_BUILTIN_MODULES: List[str] = [
    # Import paths of built-in profiles that should be available by default.
    # Keep this list tiny; it's safe to leave modules out and let them
    # import themselves when referenced from elsewhere.
    "timescales.profiles.power_law",
    "timescales.profiles.power_law_BH"
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _norm(name: str) -> str:
    """Normalize user-facing names: case-insensitive, ignore '_' and '-'."""
    return name.lower().replace("_", "").replace("-", "").strip()

def _canonical(name: str) -> Optional[str]:
    """Resolve aliases -> canonical, else return canonical if directly registered."""
    n = _norm(name)
    if n in _REGISTRY:
        return n
    return _ALIASES.get(n)

def _register_name(name: str, cls: Type[ProfileBase]) -> None:
    n = _norm(name)
    if n in _REGISTRY and _REGISTRY[n] is not cls:
        raise ValueError(f'Profile name conflict: "{name}" already registered to {_REGISTRY[n].__name__}')
    _REGISTRY[n] = cls

def _register_alias(alias: str, canonical: str) -> None:
    a = _norm(alias)
    c = _norm(canonical)
    if a in _REGISTRY:
        raise ValueError(f'Cannot alias "{alias}" because it is already a canonical profile name.')
    existing = _ALIASES.get(a)
    if existing and existing != c:
        raise ValueError(f'Alias conflict: "{alias}" -> {existing} (existing) vs {c} (new)')
    _ALIASES[a] = c

# -----------------------------------------------------------------------------
# Public registration decorator
# -----------------------------------------------------------------------------
def register_profile(name: str, *, aliases: Iterable[str] = ()) -> Callable[[Type[T]], Type[T]]:
    """
    Class decorator to register a ProfileBase subclass under a name (and aliases).

    Usage in a profile module:
        @register_profile("powerlaw", aliases=("pl", "power-law"))
        class PowerLawProfile(ProfileBase):
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, ProfileBase):
            raise TypeError(f"@register_profile can only be used on ProfileBase subclasses (got {cls})")
        _register_name(name, cls)
        for a in aliases:
            _register_alias(a, name)
        # Optional: let the class know its canonical registry name
        setattr(cls, "_registry_name", _norm(name))
        return cls
    return decorator

# -----------------------------------------------------------------------------
# Discovery (optional)
# -----------------------------------------------------------------------------
def _import_builtins() -> None:
    """Import built-in profile modules so their decorators run and register classes."""
    for mod in _BUILTIN_MODULES:
        try:
            import_module(mod)
        except Exception:
            # Keep factory robust even if some optional profiles aren't present yet.
            # You can log this if you add a logger.
            pass

# def _load_plugins() -> None:
#     """
#     Discover third-party profiles via entry points.
#     Package authors can expose profiles by declaring in their pyproject.toml:

#         [project.entry-points."timescales.profiles"]
#         nfw = "somepkg.profiles.nfw:NFWProfile"

#     The entry point name becomes the canonical registry name.
#     """
#     try:
#         eps = metadata.entry_points()  # Py>=3.10 returns EntryPoints object
#     except Exception:
#         return
#     for ep in eps.select(group="timescales.profiles"):
#         try:
#             cls = ep.load()
#             if not issubclass(cls, ProfileBase):
#                 continue
#             _register_name(ep.name, cls)
#             # Optional: let plugin provide its own aliases via a class attribute
#             aliases = getattr(cls, "ALIASES", ())
#             for a in aliases:
#                 _register_alias(a, ep.name)
#         except Exception:
#             # Ignore broken plugins; optionally log
#             continue

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def available_profiles(*, include_aliases: bool = False) -> List[str]:
    """
    Return a sorted list of available profile names.
    Set include_aliases=True to include alias names too.
    """
    # Ensure built-ins and plugins are visible
    _import_builtins()
    # _load_plugins()
    names = set(_REGISTRY.keys())
    if include_aliases:
        names |= set(_ALIASES.keys())
    return sorted(names)

def get_profile_class(name: str) -> Type[ProfileBase]:
    """
    Return the profile class for a given name/alias.
    Raises a helpful error if not found.
    """
    # Lazy discovery on demand (cheap)
    _import_builtins()
    # _load_plugins()

    canonical = _canonical(name)
    if canonical and canonical in _REGISTRY:
        return _REGISTRY[canonical]

    # Not found: build a helpful message with suggestions
    import difflib
    candidates = available_profiles(include_aliases=True)
    suggestions = difflib.get_close_matches(_norm(name), candidates, n=5, cutoff=0.5)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise KeyError(f'Unknown profile "{name}". Known profiles: {", ".join(sorted(_REGISTRY.keys()))}.{hint}')

def create_profile(name: str, /, **kwargs) -> ProfileBase:
    """
    Instantiate a registered profile by name (or alias).
    Extra kwargs are forwarded to the profile class constructor.
    """
    cls = get_profile_class(name)
    return cls(**kwargs)

available_profiles()