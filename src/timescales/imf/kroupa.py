# src/timescales/physics/kroupa.py (or wherever you keep IMF models)
from __future__ import annotations
import numpy as np
import astropy.units as u
from astropy.units import Quantity

from .registry import register_imf
from .base import imfBase

@register_imf("kroupa", aliases = ("kp"))
class kroupaIMF(imfBase):
    """
    Piecewise power-law Kroupa IMF (Kroupa 2001).

    dN/dM = k_i * M^{-alpha_i}  on each segment, with continuity at the breaks.

    Default (2 segments):
      - alpha1 = 1.3 for M in [0.08, 0.5] Msun
      - alpha2 = 2.3 for M in [0.5, 100] Msun

    Optional very-low-mass segment (3 segments):
      - alpha0 = 0.3 for M in [0.01, 0.08] Msun  (set include_low=True)

    Normalization: call get_normalization(Mtot) to set coefficients so that
    ∫ M (dN/dM) dM over [Mmin, Mmax] equals Mtot.
    """

    def __init__(
        self,
        *,
        Mmin: Quantity = 0.08 * u.Msun,
        Mmax: Quantity = 100 * u.Msun,
        breaks: tuple[Quantity, ...] = (0.5 * u.Msun,),
        alphas: tuple[float, ...] = (1.3, 2.3),
        include_low: bool = False,
        low_break: Quantity = 0.08 * u.Msun,
        low_alpha: float = 0.3,
    ):
        """
        Parameters
        ----------
        Mmin, Mmax : Quantity
            Support of the IMF.
        breaks : tuple
            Interior break masses (monotonic). For 2-seg default: (0.5 Msun,).
        alphas : tuple
            Power-law slopes per segment (same length as segments). For 2-seg default: (1.3, 2.3).
        include_low : bool
            If True, prepend a very-low-mass segment [0.01, 0.08] Msun with slope low_alpha.
        low_break : Quantity
            Upper edge of the optional very-low-mass segment (default 0.08 Msun).
        low_alpha : float
            Slope for the optional very-low-mass segment.
        """
        # Build segment edges
        edges = [Mmin]
        if include_low:
            # Insert [Mmin, low_break] as the first segment and shift others up.
            edges = [Mmin, low_break]
            alphas = (low_alpha,) + alphas
            breaks = (low_break,) + tuple(breaks)
        edges += [*breaks, Mmax]
        self.edges: list[Quantity] = [e.to(u.Msun) for e in edges]
        self.alphas: list[float] = list(alphas)
        if len(self.alphas) != len(self.edges) - 1:
            raise ValueError("Number of alphas must equal number of segments (len(edges)-1).")

        # Store and init normalization constants (k_i)
        self.Mmin = self.edges[0]
        self.Mmax = self.edges[-1]
        self._k: list[Quantity] = [None] * len(self.alphas)  # filled by get_normalization
        self._A: Quantity | None = None  # overall scale (k_0); useful to inspect

    # ---------- utilities ----------

    @staticmethod
    def _I_power(a: Quantity, b: Quantity, alpha: float) -> Quantity:
        """∫ M^{-alpha} dM from a to b (for number counts)."""
        e = 1.0 - alpha
        if np.isclose(e, 0.0, atol=1e-14):
            return np.log(b / a)
        return (b**e - a**e) / e

    @staticmethod
    def _I_mass(a: Quantity, b: Quantity, alpha: float) -> Quantity:
        """∫ M * M^{-alpha} dM = ∫ M^{1-alpha} dM (for mass)."""
        e = 2.0 - alpha
        if np.isclose(e, 0.0, atol=1e-14):
            return np.log(b / a)
        return (b**e - a**e) / e

    def _coeffs_from_A(self, A: Quantity) -> list[Quantity]:
        """
        Build piecewise coefficients k_i ensuring continuity at each break:
            k_{i+1} = k_i * (M_break_i)^{alpha_{i+1} - alpha_i}
        """
        ks = [None] * len(self.alphas)
        ks[0] = A
        for i in range(len(self.alphas) - 1):
            Mb = self.edges[i + 1]  # break mass between seg i and i+1
            ks[i + 1] = ks[i] * (Mb ** (self.alphas[i + 1] - self.alphas[i]))
        return ks

    # ---------- core API ----------

    def get_normalization(self, Mtot: Quantity) -> Quantity:
        """
        Determine normalization constants so that total *mass* equals Mtot:
            Mtot = Σ_i ∫_{M_i}^{M_{i+1}} M * (k_i M^{-alpha_i}) dM
                 = A * Σ_i C_i * ∫ M^{1 - alpha_i} dM
          with continuity setting k_i = A * C_i.

        Returns
        -------
        A : Quantity (1/Msun)
            The base coefficient (for the first segment).
        """
        # Compute continuity multipliers C_i (C_0 = 1)
        C = [1.0 * u.one]
        for i in range(len(self.alphas) - 1):
            Mb = self.edges[i + 1]
            C.append(C[-1] * (Mb ** (self.alphas[i + 1] - self.alphas[i])))

        # Total mass integral is A times the weighted sum over segments
        S = 0 * u.Msun  # integral has mass units
        for i, alpha in enumerate(self.alphas):
            a, b = self.edges[i], self.edges[i + 1]
            S += C[i] * self._I_mass(a, b, alpha)

        A = (Mtot / S).to(1 / u.Msun)
        self._A = A
        self._k = self._coeffs_from_A(A)
        return A

    def dNdM(self, M: Quantity) -> Quantity:
        """
        Differential number spectrum (normalized if get_normalization() was called).
        Returns Quantity with units 1/Msun.
        """
        if self._k[0] is None:
            raise RuntimeError("IMF not normalized. Call get_normalization(Mtot) first.")
        M = M.to(self.Mmin.unit)
        out = np.zeros_like(M.value) / u.Msun
        # piecewise apply k_i * M^{-alpha_i}
        for i, alpha in enumerate(self.alphas):
            a, b = self.edges[i], self.edges[i + 1]
            mask = (M >= a) & (M < b) if i < len(self.alphas) - 1 else (M >= a) & (M <= b)
            if np.any(mask):
                out[mask] = (self._k[i] * (M[mask] ** (-alpha))).to(1 / u.Msun)
        return out

    def dNdlogM(self, M: Quantity) -> Quantity:
        """dN/dlogM = M * dN/dM"""
        M = M.to(self.Mmin.unit)
        return (M * self.dNdM(M)).to(u.one)  # unitless count per log interval

    # ---------- fractions (by number / by mass) ----------

    def number_fraction(self, M1: Quantity, M2: Quantity) -> float:
        """
        Fraction of *stars by number* in [M1, M2] relative to the total IMF.
        Independent of normalization.
        """
        lo = np.maximum(np.minimum(M1, M2), self.Mmin).to(self.Mmin.unit)
        hi = np.minimum(np.maximum(M1, M2), self.Mmax).to(self.Mmin.unit)
        if hi <= lo:
            return 0.0

        def seg_sum(fun) -> Quantity:
            s = 0 * u.one
            for i, alpha in enumerate(self.alphas):
                a, b = self.edges[i], self.edges[i + 1]
                aa = np.maximum(a, lo); bb = np.minimum(b, hi)
                if bb > aa:
                    s += (self._I_power(aa, bb, alpha) *
                          (a**0 / a**0))  # keep as Quantity
            return s

        num = seg_sum(self._I_power)
        den = 0 * u.one
        for i, alpha in enumerate(self.alphas):
            den += self._I_power(self.edges[i], self.edges[i + 1], alpha)
        return (num / den).to_value(u.one)

    def mass_fraction(self, M1: Quantity, M2: Quantity) -> float:
        """
        Fraction of *total mass* in [M1, M2] relative to the total IMF mass.
        Independent of normalization.
        """
        lo = np.maximum(np.minimum(M1, M2), self.Mmin).to(self.Mmin.unit)
        hi = np.minimum(np.maximum(M1, M2), self.Mmax).to(self.Mmin.unit)
        if hi <= lo:
            return 0.0

        def seg_sum(fun) -> Quantity:
            s = 0 * u.Msun
            for i, alpha in enumerate(self.alphas):
                a, b = self.edges[i], self.edges[i + 1]
                aa = np.maximum(a, lo); bb = np.minimum(b, hi)
                if bb > aa:
                    s += self._I_mass(aa, bb, alpha)
            return s

        num = seg_sum(self._I_mass)
        den = 0 * u.Msun
        for i, alpha in enumerate(self.alphas):
            den += self._I_mass(self.edges[i], self.edges[i + 1], alpha)
        return (num / den).to_value(u.one)
