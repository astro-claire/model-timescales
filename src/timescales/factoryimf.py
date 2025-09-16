#src/timescales/factoryimf.py
"""
IMF factory & registry.

- imf register themselves via @register_imf("salpeter", aliases=("sp",))
- Users (and the API layer) ask for imf by name via create_imf("powerlaw", **cfg)
- available_imf() lists what's registered.

"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Callable, Dict, Iterable, List, Optional, Type, TypeVar
import pkgutil

from .imf.base import imfBase  # your abstract interface

T = TypeVar("T", bound=imfBase)

# -----------------------------------------------------------------------------
# Internal registries
# -----------------------------------------------------------------------------
_REGISTRY: Dict[str, Type[imfBase]] = {}   # canonical_name -> class
_ALIASES: Dict[str, str] = {}                  # alias_name -> canonical_name
_BUILTIN_MODULES: List[str] = [
    # Import paths of built-in profiles that should be available by default.
    # Keep this list tiny; it's safe to leave modules out and let them
    # import themselves when referenced from elsewhere.
    "timescales.imf.salpeter"
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

def _register_name(name: str, cls: Type[imfBase]) -> None:
    n = _norm(name)
    if n in _REGISTRY and _REGISTRY[n] is not cls:
        raise ValueError(f'Profile name conflict: "{name}" already registered to {_REGISTRY[n].__name__}')
    _REGISTRY[n] = cls

def _register_alias(alias: str, canonical: str) -> None:
    a = _norm(alias)
    c = _norm(canonical)
    if a in _REGISTRY:
        raise ValueError(f'Cannot alias "{alias}" because it is already a canonical imf name.')
    existing = _ALIASES.get(a)
    if existing and existing != c:
        raise ValueError(f'Alias conflict: "{alias}" -> {existing} (existing) vs {c} (new)')
    _ALIASES[a] = c

# -----------------------------------------------------------------------------
# Public registration decorator
# -----------------------------------------------------------------------------
def register_imf(name: str, *, aliases: Iterable[str] = ()) -> Callable[[Type[T]], Type[T]]:
    """
    Class decorator to register a ProfileBase subclass under a name (and aliases).

    Usage in a profile module:
        @register_imf("salpeter", aliases=("sp"))
        class salpeterIMF(imfBase):
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, imfBase):
            raise TypeError(f"@register_imf can only be used on imfBase subclasses (got {cls})")
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
    # Import anything inside timescales.profiles so decorators run
    import timescales.imf as _pkg
    for m in pkgutil.iter_modules(_pkg.__path__, _pkg.__name__ + "."):
        try:
            import_module(m.name)
        except Exception:
            # optionally log; don't crash discovery if one profile is broken
            pass
# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def available_imfs(*, include_aliases: bool = False) -> List[str]:
    """
    Return a sorted list of available imf names.
    Set include_aliases=True to include alias names too.
    """
    # Ensure built-ins and plugins are visible
    _import_builtins()
    # _load_plugins()
    names = set(_REGISTRY.keys())
    if include_aliases:
        names |= set(_ALIASES.keys())
    return sorted(names)

def get_imf_class(name: str) -> Type[imfBase]:
    """
    Return the imf class for a given name/alias.
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
    candidates = available_imfs(include_aliases=True)
    suggestions = difflib.get_close_matches(_norm(name), candidates, n=5, cutoff=0.5)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise KeyError(f'Unknown imf "{name}". Known imf: {", ".join(sorted(_REGISTRY.keys()))}.{hint}')

def create_imf(name: str, /, **kwargs) -> imfBase:
    """
    Instantiate a registered imf by name (or alias).
    Extra kwargs are forwarded to the profile class constructor.
    """
    cls = get_imf_class(name)
    return cls(**kwargs)

