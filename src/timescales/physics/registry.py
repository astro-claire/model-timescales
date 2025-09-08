#src/timescales/physics/registry.py
"""
Registry for timescale calculator functions in `timescales.physics`.

Usage:
    from timescales.physics.registry import register_timescale, get_timescale, available_timescales

    @register_timescale("t_coll", aliases=("collision",))
    def collision_timescale(*, n, v, m_star, **kwargs) -> "Quantity":
        ...

`analysis` code can then call:
    f = get_timescale("t_coll"); t = f(n=n, v=sigma, m_star=Mstar)
"""

from __future__ import annotations
from importlib import import_module, metadata
from typing import Callable, Dict, Iterable, List, Optional
import pkgutil

_REGISTRY: Dict[str, Callable] = {}   # canonical name -> function
_ALIASES: Dict[str, str] = {}         # alias -> canonical

def _norm(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "").strip()

def _register_name(name: str, fn: Callable) -> None:
    n = _norm(name)
    if n in _REGISTRY and _REGISTRY[n] is not fn:
        raise ValueError(f'Timescale name conflict: "{name}" already registered.')
    _REGISTRY[n] = fn

def _register_alias(alias: str, canonical: str) -> None:
    a, c = _norm(alias), _norm(canonical)
    if a == c:
        return
    if a in _REGISTRY:
        raise ValueError(f'Alias "{alias}" collides with a canonical name.')
    existing = _ALIASES.get(a)
    if existing and existing != c:
        raise ValueError(f'Alias conflict: "{alias}" -> {existing} vs {c}')
    _ALIASES[a] = c

def register_timescale(name: str, *, aliases: Iterable[str] = ()) -> Callable[[Callable], Callable]:
    """
    Decorator to register a timescale calculator function.

    The function should be a pure callable that returns an astropy Quantity.
    It may accept keyword-only inputs (e.g., n, v, rho, m_star, etc.).
    """
    def deco(fn: Callable) -> Callable:
        _register_name(name, fn)
        for a in aliases:
            _register_alias(a, name)
        setattr(fn, "_registry_name", _norm(name))
        return fn
    return deco

# ---------- Discovery ----------

def _import_physics_modules() -> None:
    """Import all modules under timescales.physics so decorators run."""
    import timescales.physics as _pkg
    for mod in pkgutil.iter_modules(_pkg.__path__, _pkg.__name__ + "."):
        try:
            import_module(mod.name)
        except Exception:
            # Optional: log
            pass

# def _load_plugins() -> None:
#     """Discover third-party calculators via entry points (optional)."""
#     try:
#         eps = metadata.entry_points()
#     except Exception:
#         return
#     for ep in eps.select(group="timescales.calculators"):
#         try:
#             fn = ep.load()
#             if callable(fn):
#                 _register_name(ep.name, fn)
#                 # Optional aliases from attribute
#                 for a in getattr(fn, "ALIASES", ()):
#                     _register_alias(a, ep.name)
#         except Exception:
#             pass

# ---------- Public API ----------

def available_timescales(*, include_aliases: bool = False) -> List[str]:
    _import_physics_modules()
    # _load_plugins()
    names = set(_REGISTRY.keys())
    if include_aliases:
        names |= set(_ALIASES.keys())
    return sorted(names)

def get_timescale(name: str) -> Callable:
    _import_physics_modules()
    _load_plugins()
    key = _norm(name)
    canon = key if key in _REGISTRY else _ALIASES.get(key)
    if canon is None or canon not in _REGISTRY:
        import difflib
        sugg = difflib.get_close_matches(key, available_timescales(include_aliases=True), n=5, cutoff=0.5)
        hint = f" Did you mean: {', '.join(sugg)}?" if sugg else ""
        raise KeyError(f'Unknown timescale "{name}".{hint}')
    return _REGISTRY[canon]

