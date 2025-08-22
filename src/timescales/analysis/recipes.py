"""
High-level analysis workflows built on `tables.py`.
Each recipe returns tidy data (and optional summaries) for easy plotting/comparison.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
import astropy.units as u
from astropy.units import Quantity

from .tables import structural_table, timescale_table, get_system
from ..physics.stars import main_sequence_lifetime_approximation  # when you add it



def collision_vs_main_sequence(
    ensemble, *,
    m_star: Optional[Quantity]=None,
    # Either provide a callable for t_MS, or later wire to physics.stars.main_sequence_lifetime
    t_ms_func = main_sequence_lifetime_approximation,                                # signature: t_ms_func(m_star: Quantity, **kwargs) -> Quantity
    t_ms_kwargs: Optional[Dict] = None,
    coulomb_log: float = 10.0,
    collision_kwargs: Optional[Dict] = None,
    system_ids: Optional[Iterable[Union[int, str]]] = None,
    return_: Literal["table", "summary", "both"] = "table",
    as_: Literal["dict", "pandas"] = "dict",
    ) -> Union[Dict, Tuple[Dict, Dict], "pandas.DataFrame"]:
    """
    Compare stellar main-sequence lifetime with stellar collision timescale
    as a function of radius for every system, and identify where collisions dominate.

    Parameters
    ----------
    ensemble
        TimescaleEnsemble supplying structural fields per system.
    m_star
        Characteristic stellar mass used for n(r) and cross-section.
    t_ms_func
        Function that returns a main-sequence lifetime for `m_star`. You can later
        replace this with a wrapper around physics.stars.main_sequence_lifetime(...).
    t_ms_kwargs
        Extra args for t_ms_func (e.g., metallicity).
    coulomb_log
        ln Λ for relaxation time (not used directly here unless you also compute t_relax).
    collision_kwargs
        Extra args for collision_timescale (eccentricity, Mcollisions, etc.).
    return_
        - "table": tidy per-(system, r) results with columns:
        TODO: (none of this implemented yet)
            ["system_id", "r", "t_coll", "t_ms", "collisions_faster" (bool)]
        - "summary": per-system scalars (e.g., crossover radius, min ratio)
        - "both": return (table, summary)
    as_
        "dict" or "pandas" (only applies when return_ includes "table").

    Returns
    -------
    If return_ == "table":
        Table/DF with per-(system, r) results.
    If return_ == "summary":
        Dict keyed by system_id with:
            - "any_collisions_faster" : bool
            - "min_ratio_tcoll_over_tms" : float
            - "crossover_radii" : Quantity list (r where t_coll == t_ms within tolerance)
            - "fraction_of_radii_collisions_faster" : float in [0,1]
    If return_ == "both":
        (table, summary)

    Method
    ------
    1) Build t_coll(r) via `timescale_table(...)`.
    2) Compute scalar t_ms = t_ms_func(m_star, **t_ms_kwargs) for the chosen stellar mass.
    3) Compare: collisions_faster(r) = [t_coll(r) < t_ms].
    4) TODO Optional: locate crossover radii where t_coll ~ t_ms (by sign changes or interpolation).
    """
    N = len(ensemble.radii)
    ids = list(range(N)) if system_ids is None else list(system_ids)
    if len(ids) != N:
        raise ValueError("Length of system_ids must match number of systems.")

    if m_star is None:
        if  hasattr(ensemble, "Mstar"):
            print("Using ensemble value of Mstar: "+str(ensemble.Mstar))
            m_star = ensemble.Mstar
        else:
            raise AttributeError("ensemble has no attribute Mstar. Provide mass as keyword argument")

    out = timescale_table(ensemble, include = ("t_coll"), m_star = m_star, collision_kwargs = collision_kwargs,system_ids = system_ids)
    
    
    if t_ms_kwargs:
        t_ms = t_ms_func(m_star, **t_ms_kwargs)
    else:
        t_ms = t_ms_func(m_star)

    out["tms/tcoll"]=[]

    for sys_id, r in zip(ids, ensemble.radii):
        sys_data = get_system(out,sys_id)
        for j in range(len(r)):
            ratio_j = t_ms/sys_data["t_coll"][j]
            out["tms/tcoll"].append(ratio_j)

    if as_ == "dict":
        return out

    # Optional pandas return
    try:
        import pandas as pd  # local import to keep pandas optional
    except Exception as e:
        raise ImportError('pandas is required when as_="pandas". Install with `pip install pandas`.') from e

    # Build a DataFrame without stripping units (object dtype for Quantity columns)
    return pd.DataFrame(out)


