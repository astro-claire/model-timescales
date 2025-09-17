"""
High-level analysis workflows built on `tables.py`.
Each recipe returns tidy data (and optional summaries) for easy plotting/comparison.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
import astropy.units as u
from astropy.units import Quantity
import warnings
from .tables import structural_table, timescale_table
from .tools import get_system
from ..physics.stars import main_sequence_lifetime_approximation, stellar_radius_approximation  # when you add it
from ..utils.energy import escape_velocity
from .tools import condition_test

def collision_vs_main_sequence(
    ensemble, *,
    # Either provide a callable for t_MS, or later wire to physics.stars.main_sequence_lifetime
    t_ms_func = main_sequence_lifetime_approximation,                                # signature: t_ms_func(m_star: Quantity, **kwargs) -> Quantity
    t_ms_kwargs: Optional[Dict] = None,
    system_ids: Optional[Iterable[Union[int, str]]] = None,
    return_: Literal["table", "summary", "both"] = "table",
    as_: Literal["dict", "pandas"] = "dict",
    verbose = True, 
    m_star: Optional[Quantity]=None, #deprecated
    coulomb_log=  None, #deprecated
    collision_kwargs: Optional[Dict] = None, #deprecated
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
    if not m_star == None: 
        if verbose:
            warnings.warn("DeprecationWarning: Adding m_star as keyword argument is deprecated. Initiate ensemble with correct mstar.")
    if not coulomb_log == None:
        if verbose: 
            warnings.warn("DeprecationWarning: Adding coulomb_log as keyword argument is deprecated. Initiate ensemble with correct coulomb_log in timescale kwargs.")
    if not collision_kwargs == None:
        if verbose: 
            warnings.warn("DeprecationWarning: Adding collision_kwargs as keyword argument is deprecated. Initiate ensemble with correct collision_kwargs in timescales_kwargs.")


    N = len(ensemble.radii)
    ids = list(range(N)) if system_ids is None else list(system_ids)
    if len(ids) != N:
        raise ValueError("Length of system_ids must match number of systems.")

    if  hasattr(ensemble, "Mstar"):
        m_star = ensemble.Mstar
    elif "Mstar" in ensemble.timescales_kwargs:
        m_star  = ensemble.timescales_kwargs['Mstar']
    else:
        raise AttributeError("ensemble has no attribute Mstar.")


    out = timescale_table(ensemble, 
                            include = ("t_coll"),
                            system_ids = system_ids, 
                            verbose = verbose)
    
    
    if t_ms_kwargs:
        t_ms = t_ms_func(m_star, **t_ms_kwargs).to('yr')
    else:
        t_ms = t_ms_func(m_star).to('yr')

    out["tms/tcoll"]=[]
    out["collisions"]=[] #will store 1 if collisions important, 0 if not

    for sys_id, r in zip(ids, ensemble.radii):
        sys_data = get_system(out,sys_id)
        for j in range(len(r)):
            ratio_j = t_ms/(sys_data["t_coll"][j]).to('yr')
            out["tms/tcoll"].append(ratio_j)
            if ratio_j >1.:
                out["collisions"].append(1.)
            else: 
                out["collisions"].append(0.)

    if as_ == "dict":
        return out

    # Optional pandas return
    try:
        import pandas as pd  # local import to keep pandas optional
    except Exception as e:
        raise ImportError('pandas is required when as_="pandas". Install with `pip install pandas`.') from e

    # Build a DataFrame without stripping units (object dtype for Quantity columns)
    return pd.DataFrame(out)


def destructive_colllision_criterion(
    ensemble, *,
    r_ms_function = stellar_radius_approximation,
    r_ms_kwargs: Optional[Dict] = None,
    system_ids: Optional[Iterable[Union[int, str]]] = None,
    as_: Literal["dict", "pandas"] = "dict",
    verbose = True, 
    m_star: Optional[Quanity] = None, #deprecated
) -> Union[Table, "pandas.DataFrame"]:
    """
    Build a per (system, radius) table of whether collisions should be considered constructive or destructive
    
    Parameters
    ----------
    ensemble
        A `TimescaleEnsemble` instance with .radii and corresponding model profiles.
    r_ms_function
        Function to calculate the stellar radius based off the stellar mass. 
            Default: stellar_radius_approximation
    r_ms_kwargs
        Optional keyword arguments for r_ms_function
            Default: None
    m_star
        Deprecated
    system_ids
        Optional explicit labels for systems; else indices (0..N-1) are used.
    as_
        "dict" → return a columnar dict of lists of Quantities.
        "pandas" → return a DataFrame (requires pandas installed).

    Returns
    -------
    Table or DataFrame with columns:
        - "system_id"     (int/str)
        - "r"             (length Quantity)
        - "sigma/vesc" (dimensionless Quantity)
        - "massloss" (int)
    """
    if not m_star == None: 
        if verbose:
            warnings.warn("DeprecationWarning: Adding m_star as keyword argument is deprecated. Initiate ensemble with correct mstar.")
    
    N = len(ensemble.radii)
    ids = list(range(N)) if system_ids is None else list(system_ids)
    if len(ids) != N:
        raise ValueError("Length of system_ids must match number of systems.")

    if  hasattr(ensemble, "Mstar"):
        m_star = ensemble.Mstar
    elif "Mstar" in ensemble.timescales_kwargs:
        m_star  = ensemble.timescales_kwargs['Mstar']
    else:
        raise AttributeError("ensemble has no attribute Mstar.")

    out = structural_table(ensemble, 
                        fields = ("sigma"), 
                        system_ids=system_ids, 
                        )
    
    if r_ms_kwargs:
        r_ms = r_ms_function(m_star, **r_ms_kwargs)
    else:
        r_ms = r_ms_function(m_star)

    v_esc = escape_velocity(m_star, r_ms).to(out["sigma"][0].unit)

    out["sigma/vesc"]=[]
    out["massloss"]=[] # 1 if mass loss, 0 if not

    for sys_id, r in zip(ids, ensemble.radii):
        sys_data = get_system(out,sys_id)
        for j in range(len(r)):
            ratio_j = sys_data["sigma"][j]/v_esc
            out["sigma/vesc"].append(ratio_j)
            if ratio_j >1.:
                out["massloss"].append(1.)
            else: 
                out["massloss"].append(0.)

    if as_ == "dict":
        return out

    # Optional pandas return
    try:
        import pandas as pd  # local import to keep pandas optional
    except Exception as e:
        raise ImportError('pandas is required when as_="pandas". Install with `pip install pandas`.') from e

    # Build a DataFrame without stripping units (object dtype for Quantity columns)
    return pd.DataFrame(out)

def generate_timescale_comparison(
    ensemble, *,
    include: Iterable[str] = ("t_relax", "t_coll" ,"t_ms" , "massloss"),
    r_ms_function = stellar_radius_approximation,
    r_ms_kwargs: Optional[Dict] = None,
    t_ms_func = main_sequence_lifetime_approximation,                                # signature: t_ms_func(m_star: Quantity, **kwargs) -> Quantity
    t_ms_kwargs: Optional[Dict] = None,
    system_ids: Optional[Iterable[Union[int, str]]] = None,
    as_: Literal["dict", "pandas"] = "dict",
    verbose = True,
    m_star: Optional[Quantity]=None, #deprecated
    coulomb_log: float = None, #deprecated
    collision_kwargs: Optional[Dict] = None, #deprecated
) -> Union[Table, "pandas.DataFrame"]:
    """ 
    Compares all star cluster timescales at each radius

    Parameters
    ----------
    ensemble
        A `TimescaleEnsemble` instance with .radii and corresponding model profiles.
    r_ms_function
        Function to calculate the stellar radius based off the stellar mass. 
            Default: stellar_radius_approximation
    r_ms_kwargs
        Optional keyword arguments for r_ms_function
            Default: None
    t_ms_func
        Function that returns a main-sequence lifetime for `m_star`. You can later
        replace this with a wrapper around physics.stars.main_sequence_lifetime(...).
    t_ms_kwargs
        Extra args for t_ms_func (e.g., metallicity).
    coulomb_log
        ln Λ for relaxation time (not used directly here unless you also compute t_relax).
    collision_kwargs
        Extra args for collision_timescale (eccentricity, Mcollisions, etc.).
    m_star
        Stellar mass for number density if desired to be different from ensemble's.
    system_ids
        Optional explicit labels for systems; else indices (0..N-1) are used.
    as_
        "dict" → return a columnar dict of lists of Quantities.
        "pandas" → return a DataFrame (requires pandas installed).
    """
    if not m_star == None: 
        if verbose:
            warnings.warn("DeprecationWarning: Adding m_star as keyword argument is deprecated. Initiate ensemble with correct mstar.")
    if not coulomb_log == None:
        if verbose: 
            warnings.warn("DeprecationWarning: Adding coulomb_log as keyword argument is deprecated. Initiate ensemble with correct coulomb_log in timescale kwargs.")
    if not collision_kwargs == None:
        if verbose: 
            warnings.warn("DeprecationWarning: Adding collision_kwargs as keyword argument is deprecated. Initiate ensemble with correct collision_kwargs in timescales_kwargs.")

    N = len(ensemble.radii)
    ids = list(range(N)) if system_ids is None else list(system_ids)
    if len(ids) != N:
        raise ValueError("Length of system_ids must match number of systems.")

    if  hasattr(ensemble, "Mstar"):
        m_star = ensemble.Mstar
    elif "Mstar" in ensemble.timescales_kwargs:
        m_star  = ensemble.timescales_kwargs['Mstar']
    else:
        raise AttributeError("ensemble has no attribute Mstar.")

    out = timescale_table(ensemble, 
                    include = include, 
                    system_ids = system_ids,
                    verbose=verbose)

    if "t_ms" in include:
        if t_ms_kwargs:
            t_ms = t_ms_func(m_star, **t_ms_kwargs).to('yr')
        else:
            t_ms = t_ms_func(m_star).to('yr')
    include = list(include)
    if "massloss" in include:
        masslosstable = destructive_colllision_criterion(ensemble,
                                                        r_ms_function=r_ms_function,
                                                        r_ms_kwargs = r_ms_kwargs,
                                                        system_ids = system_ids,
                                                        )
        out["sigma/vesc"] = masslosstable["sigma/vesc"]
        out["massloss"] = masslosstable["massloss"]
        include.remove("massloss")
    if "t_relax" and "t_coll" in include:
        out["t_relax/t_coll"]=[]
    if "t_relax" and "t_ms" in include:
        out["t_ms/t_relax"] = []
    if "t_coll" and "t_ms" in include: 
        out["t_ms/t_coll"] =[]

    out["shortest"]=[]
    if "t_ms" in include: 
        out["t_ms"] =[]

    if len(include)>1:
        for sys_id, r in zip(ids, ensemble.radii):
            sys_data = get_system(out,sys_id)
            for j in range(len(r)):
                if "t_ms" in include: 
                    shortest_name = "t_ms"
                    shortest_tscale = t_ms
                    out['t_ms'].append(t_ms)
                else: 
                    shortest_name = "dummy"
                    shortest_tscale = 1e50*u.yr # should be too long to appear
                if "t_relax" in include and sys_data["t_relax"][j]<shortest_tscale:
                    shortest_name = "t_relax"
                    shortest_tscale = sys_data["t_relax"][j]
                if "t_coll" in include and sys_data["t_coll"][j]<shortest_tscale:
                    shortest_name = "t_coll"
                    shortest_tscale = sys_data["t_coll"][j]
                out["shortest"].append(shortest_name)
                if "t_relax" and "t_coll" in include: 
                    out["t_relax/t_coll"].append(sys_data["t_relax"][j]/sys_data["t_coll"][j])
                if "t_relax" and "t_ms" in include:
                    out["t_ms/t_relax"].append(t_ms/sys_data["t_relax"][j])
                if "t_coll" and "t_ms" in include: 
                    out["t_ms/t_coll"].append(t_ms/sys_data["t_coll"][j])

# ("t_relax", "t_coll" ,"t_ms" , "massloss"),

    
    if as_ == "dict":
        return out

    # Optional pandas return
    try:
        import pandas as pd  # local import to keep pandas optional
    except Exception as e:
        raise ImportError('pandas is required when as_="pandas". Install with `pip install pandas`.') from e

    # Build a DataFrame without stripping units (object dtype for Quantity columns)
    return pd.DataFrame(out)

def per_system_comparison(table,ts1_name, operation,*,
                        ts2_name = None,
                        value = None,
                        system_ids: Optional[Iterable[Union[int, str]]] = None,
                        as_: Literal["dict", "pandas"] = "dict",):
    """
    check whether each system meets a certain criterion anywhere in the system
    """
    # N = len(set(table['system_id']))
    # ids = list(range(N)) if system_ids is None else list(system_ids)
    out: Table = {
        "system_id": [],  # -> list[str|int], fine to store as plain python scalars
        "condition": [],          # -> Quantity (length)
    }
    as_df = False
    if as_ =="pandas":
        as_df = True 
        N  = len(set(table['system_id']))
        ids = pandas.unique(table["system_id"])
    else:
        N = len(set(table['system_id']))
        ids = list(range(N)) if system_ids is None else list(system_ids)
    for sys_id in ids:
        sys_data = get_system(table,sys_id, as_df=as_df)
        if operation != 'true':
            if ts2_name != None: 
                condition_value = condition_test(sys_data[ts1_name],operation,ts2 = sys_data[ts2_name])
            elif value != None: 
                try:
                    condition_value = condition_test(sys_data[ts1_name],operation,value = value)
                except TypeError:
                    condition_value = condition_test(sys_data[ts1_name],operation,value = value[sys_id])
            else:
                raise TypeError(str(operation)+" requested but no comparison ts or value given.")
        elif operation== 'true':
            condition_value = condition_test(sys_data[ts1_name],operation)
        out["system_id"].append(sys_id)
        out["condition"].append(condition_value)

    if as_ == "dict":
        return out

    # Optional pandas return
    try:
        import pandas as pd  # local import to keep pandas optional
    except Exception as e:
        raise ImportError('pandas is required when as_="pandas". Install with `pip install pandas`.') from e

    # Build a DataFrame without stripping units (object dtype for Quantity columns)
    return pd.DataFrame(out)