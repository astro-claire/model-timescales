# src/timescales/profiles/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional

import numpy as np
import astropy.units as u
from astropy.units import Quantity

__all__ = ["ProfileBase"]


class ProfileBase(ABC):
    """
    Abstract base class for structural profiles (density/kinematics).

    Contract (for subclasses):
      - Constructor should accept model parameters (e.g., alpha, r0, r_min, etc.)
        and store them on `self` (e.g., self.alpha).
      - Methods must accept/return `astropy.units.Quantity` with appropriate units.
      - Methods should be vectorized over `r` (1D Quantity array). Shape in == shape out.
      - No global I/O or stateful side effects; pure functions of (r, params).

    Notes on usage:
      - The API/orchestrator will typically loop over systems and call these methods
        with that system's radial grid (ragged arrays are fine at the API layer).
      - Keep conversions/validation minimal here; centralize generic checks in utils.
    """

    #: Optional: subclasses can define a tuple of alias strings (used by the factory).
    ALIASES: tuple[str, ...] = ()

    def __init__(self, **kwargs: Any) -> None:
        """
        Base ctor keeps kwargs for transparency; subclasses define explicit params.
        Example in a subclass:
            def __init__(self, alpha: float = 2.0, r0: Quantity | None = None, r_min: Quantity = 0*u.cm):
                self.alpha = float(alpha)
                self.r0 = r0
                self.r_min = r_min
        """
        # Store raw init kwargs for reproducibility/serialization if desired.
        self._init_kwargs: Dict[str, Any] = dict(kwargs)

    # ------------- Introspection helpers (optional, but handy) ----------------

    @property
    def params(self) -> Mapping[str, Any]:
        """
        Return a read-only mapping of model parameters for logging/serialization.
        Subclasses may override to expose a curated dict.
        """
        return dict(self._init_kwargs)

    # ------------------------- Required interface -----------------------------

    @abstractmethod
    def density(self, r: Quantity) -> Quantity:
        """
        Volume mass density ρ(r).

        Parameters
        ----------
        r : Quantity (length), shape (N,)
            Radii at which to evaluate the density.

        Returns
        -------
        rho : Quantity (mass / length^3), shape (N,)
        """
        raise NotImplementedError

    @abstractmethod
    def enclosed_mass(self, r: Quantity) -> Quantity:
        """
        Enclosed mass M(<r).

        Parameters
        ----------
        r : Quantity (length), shape (N,)

        Returns
        -------
        M : Quantity (mass), shape (N,)
        """
        raise NotImplementedError

    @abstractmethod
    def velocity_dispersion(self, r: Quantity) -> Quantity:
        """
        One-dimensional velocity dispersion σ(r).

        Parameters
        ----------
        r : Quantity (length), shape (N,)

        Returns
        -------
        sigma : Quantity (speed), shape (N,)
        """
        raise NotImplementedError

    # ----------------------- Default/derived utilities ------------------------

    def number_density(self, r: Quantity, m_star: Quantity) -> Quantity:
        """
        Stellar number density n(r). Default assumes n = ρ / m_*.
        Override in subclasses if your profile implies a different relation.

        Parameters
        ----------
        r : Quantity (length), shape (N,)
        m_star : Quantity (mass)
            Typical stellar mass.

        Returns
        -------
        n : Quantity (1 / length^3), shape (N,)
        """
        rho = self.density(r)
        if not isinstance(m_star, u.Quantity):
            raise TypeError("m_star must be an astropy Quantity with mass units.")
        # Return in natural units of rho/m_star; caller may .to(1/u.cm**3) if desired.
        return rho / m_star

    def fields(self, r: Quantity, *, include_number_density: bool = False, m_star: Optional[Quantity] = None) -> dict[str, Quantity]:
        """
        Convenience: compute a bundle of structural fields at once.

        Parameters
        ----------
        r : Quantity (length), shape (N,)
        include_number_density : bool
            If True, also compute n(r) (requires m_star).
        m_star : Quantity (mass), optional
            Stellar mass for n(r).

        Returns
        -------
        out : dict with keys:
            - "r": r
            - "rho": density(r)
            - "Menc": enclosed_mass(r)
            - "sigma": velocity_dispersion(r)
            - (optional) "n": number_density(r, m_star)
        """
        out = {
            "r": r,
            "rho": self.density(r),
            "Menc": self.enclosed_mass(r),
            "sigma": self.velocity_dispersion(r),
        }
        if include_number_density:
            if m_star is None:
                raise ValueError("m_star is required when include_number_density=True.")
            out["n"] = self.number_density(r, m_star)
        return out
