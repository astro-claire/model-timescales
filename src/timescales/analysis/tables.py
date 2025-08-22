"""
Utilities to assemble *tabular* (tidy) results from a TimescaleEnsemble.

Design goals:
- Model-agnostic: consume only the public ensemble/profile APIs.
- Tidy orientation: (system_id, r, field) rows so comparisons are simple.
- Pandas-optional: return plain Python structures; upgrade to DataFrame if available.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
import astropy.units as u
from astropy.units import Quantity
from ..physics.collisions import collision_timescale
from ..physics.relaxation import relaxation_timescale

# Type alias for the default, Pandas-free return type
Row = Dict[str, Quantity]                      # a single row of quantities
Table = Dict[str, List[Quantity]]              # columnar lists with Quantity entries

def structural_table(
    ensemble, *,
    fields: Iterable[str] = ("rho", "sigma", "Menc"),
    include_number_density: bool = False,
    m_star: Optional[Quantity] = None,
    system_ids: Optional[Iterable[Union[int, str]]] = None,
    as_: Literal["dict", "pandas"] = "dict",
) -> Union[Table, "pandas.DataFrame"]:
    """
    Build a per-(system, radius) table of structural fields from a TimescaleEnsemble.

    Parameters
    ----------
    ensemble
        A `TimescaleEnsemble` instance with .radii and corresponding model profiles.
    fields
        Structural fields to include. Recognized: "rho", "sigma", "Menc".
    include_number_density
        If True, include column "n" computed as rho / m_star (unless a profile overrides).
    m_star
        Stellar mass for number density if include_number_density=True.
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
        - requested fields (Quantities), e.g., "rho", "sigma", "Menc"
        - optional "n"

    Notes
    -----
    - One row per (system, sampled radius).
    - Units are preserved on each column via `astropy.units.Quantity`.
    - If `as_="pandas"`, Quantity columns will be object dtype unless you strip units.
    """
    N = len(ensemble.radii)
    ids = list(range(N)) if system_ids is None else list(system_ids)
    if len(ids) != N:
        raise ValueError("Length of system_ids must match number of systems.")


    want_rho = "rho" in fields or include_number_density
    want_sigma = "sigma" in fields
    want_Menc = "Menc" in fields

    # Decide how to fetch fields: prefer precomputed lists on the ensemble, else use profiles
    has_attr = lambda name: hasattr(ensemble, name) and len(getattr(ensemble, name)) == N

    use_profiles = not (
        (not want_rho or has_attr("rho"))
        and (not want_sigma or has_attr("sigma"))
        and (not want_Menc or has_attr("Menc"))
    )

    # Prepare columns
    out: Table = {
        "system_id": [],  # -> list[str|int], fine to store as plain python scalars
        "r": [],          # -> Quantity (length)
    }
    if "rho" in fields:
        out["rho"] = []
    if "sigma" in fields:
        out["sigma"] = []
    if "Menc" in fields:
        out["Menc"] = []
    if include_number_density:
        if m_star is None:
            raise ValueError("m_star is required when include_number_density=True.")
        out["n"] = []
    
    # ---- main loop over systems (handles ragged radii) ----
    for sys_id, r in zip(ids, ensemble.radii):
        # Fetch per-system fields, either from cached attributes or from the profile
        if use_profiles:
            # Expect: ensemble.profiles[i] implements density(), enclosed_mass(), velocity_dispersion()
            i = sys_id if isinstance(sys_id, int) else ids.index(sys_id)
            prof = ensemble.profiles[i]
            rho_i = prof.density(r) if want_rho else None
            sigma_i = prof.velocity_dispersion(r) if want_sigma else None
            Menc_i = prof.enclosed_mass(r) if want_Menc else None
        else:
            rho_i = ensemble.rho[ids.index(sys_id)] if want_rho else None
            sigma_i = ensemble.sigma[ids.index(sys_id)] if want_sigma else None
            Menc_i = ensemble.Menc[ids.index(sys_id)] if want_Menc else None

        # Append one row per radius for this system
        # (r is a 1D Quantity array; index across it)
        for j in range(len(r)):
            out["system_id"].append(sys_id)
            out["r"].append(r[j])
            if "rho" in fields:
                out["rho"].append(rho_i[j])           # type: ignore[index]
            if "sigma" in fields:
                out["sigma"].append(sigma_i[j])        # type: ignore[index]
            if "Menc" in fields:
                out["Menc"].append(Menc_i[j])          # type: ignore[index]
            if include_number_density:
                # Use rho if we have it already; otherwise compute just-in-time
                rho_val = rho_i[j] if want_rho else ensemble.profiles[ids.index(sys_id)].density(r[j])
                out["n"].append((rho_val / m_star).to(1 / u.pc**3))


    if as_ == "dict":
        return out

    # Optional pandas return
    try:
        import pandas as pd  # local import to keep pandas optional
    except Exception as e:
        raise ImportError('pandas is required when as_="pandas". Install with `pip install pandas`.') from e

    # Build a DataFrame without stripping units (object dtype for Quantity columns)
    return pd.DataFrame(out)



def timescale_table(
    ensemble, *,
    include: Iterable[str] = ("t_relax", "t_coll"),
    m_star: Optional[Quantity]=None,
    coulomb_log: float = 10.0,
    collision_kwargs: Optional[Dict] = None,
    system_ids: Optional[Iterable[Union[int, str]]] = None,
    as_: Literal["dict", "pandas"] = "dict",
) -> Union[Table, "pandas.DataFrame"]:
    """
    Build a per-(system, radius) table of *timescales* computed from structural fields.

    Parameters
    ----------
    ensemble
        The TimescaleEnsemble providing r, rho(r), sigma(r), etc.
    include
        Which timescales to compute. Recognized (extend as you add physics):
            - "t_relax"  (uses relaxation_timescale)
            - "t_coll"   (uses collision_timescale)
            - "t_cross"  (optional future: crossing time)
            - "t_df"     (optional future: dynamical friction)
    m_star
        Stellar mass, needed for number density (n = rho / m_star) and collision geometry.
    coulomb_log
        ln Λ for relaxation time (passed through to physics).
    collision_kwargs
        Extra keyword args for collision model (e.g., eccentricity e, Mcollisions, etc.).
    system_ids
        Optional system labels.
    as_
        "dict" or "pandas" (see structural_table).

    Returns
    -------
    Table or DataFrame with columns:
        - "system_id", "r"
        - one column per timescale requested (Quantities with time units)

    Notes
    -----
    - Pulls rho/sigma from the profiles; computes n from rho and m_star unless overridden.
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

    want_tcoll = "t_coll" in include
    want_trelax = "t_relax" in include

    fields = []
    if want_tcoll:
        fields.append("rho")
        fields.append("sigma")
        want_n = True
    if want_trelax:
        if "sigma" not in fields:
            fields.append("sigma")
        if "rho" not in fields:
            fields.append("rho")

    fields_table = structural_table(ensemble, fields= fields,m_star = m_star, include_number_density=want_n, system_ids =system_ids)

    # Prepare columns
    out: Table = {
        "system_id": [],  # -> list[str|int], fine to store as plain python scalars
        "r": [],          # -> Quantity (length)

    }
    if want_tcoll:
        out["t_coll"]= []
    if want_trelax:
        out["t_relax"] = []


    for sys_id, r in zip(ids, ensemble.radii):
        for j in range(len(r)):
            out["system_id"].append(sys_id)
            out["r"].append(r[j])
            if want_tcoll:
                if collision_kwargs:
                    out["t_coll"].append(collision_timescale(fields_table["n"][j],
                                                        fields_table["sigma"][j],
                                                        m_star,
                                                        **collision_kwargs))
                else:
                    out["t_coll"].append(collision_timescale(fields_table["n"][j],
                                                        fields_table["sigma"][j],
                                                        m_star))
            if want_trelax:
                out["t_relax"].append(relaxation_timescale(fields_table["sigma"][j],
                                                        fields_table["rho"][j],
                                                        mass = m_star,
                                                        coulomb = coulomb_log
                                                        ))
    
    if as_ == "dict":
        return out

    # Optional pandas return
    try:
        import pandas as pd  # local import to keep pandas optional
    except Exception as e:
        raise ImportError('pandas is required when as_="pandas". Install with `pip install pandas`.') from e

    # Build a DataFrame without stripping units (object dtype for Quantity columns)
    return pd.DataFrame(out)