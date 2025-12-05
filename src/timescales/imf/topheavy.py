# src/timescales/imf/top_heavy_chon22.py
from __future__ import annotations
import numpy as np
import astropy.units as u
from astropy.units import Quantity

from .base import imfBase
from .registry import register_imf
from ..utils.units import as_quantity  # if you want to use it later


@register_imf("topheavy_chon22", aliases=("topheavy", "top-heavy","logflat", "chon22"))
class TopHeavyChon22IMF(imfBase):
    r"""
    Top-heavy, approximately log-flat IMF motivated by Chon et al. (2022, 2023).

    Functional form
    ----------------
    We assume a single-slope power-law in mass

        dN/dM ∝ M^{-α},

    with α ≈ 1 so that

        dN/dlog M = M * dN/dM ≈ const,

    i.e. a log-flat mass function over [Mmin, Mmax].

    This captures the "top-heavy" behaviour found in simulations of low-Z / high-z
    star formation where the IMF becomes significantly flatter than Salpeter 
    at M ≳ few Msun (Chon et al. 2022, 2023).

    Parameters
    ----------
    Mmin : `~astropy.units.Quantity`, optional
        Minimum stellar mass. Default is 0.1 Msun.
    Mmax : `~astropy.units.Quantity`, optional
        Maximum stellar mass. Default is 300 Msun.
    beta : float, optional
        Power-law index in dN/dM ∝ M^{-beta}. Default is 1.0 (log-flat).
    """

    def __init__(
        self,
        Mmin: Quantity = 0.1 * u.Msun,
        Mmax: Quantity = 300.0 * u.Msun,
        beta: float = 1.0,
    ):
        self.beta = float(beta)
        self.Mmin = Mmin
        self.Mmax = Mmax
        self._A = None  # normalization constant for dN/dM

    # ------------------------------------------------------------------
    # Core differentials
    # ------------------------------------------------------------------
    def dNdM(self, M: Quantity) -> Quantity:
        """
        Mass spectrum dN/dM.

        Requires normalization constant self._A to be set via `get_normalization`.
        """
        if self._A is None:
            raise RuntimeError("IMF not normalized yet. Call get_normalization first.")
        M = M.to(self.Mmin.unit)
        return self._A * (M ** (-self.beta))

    def dNdlogM(self, M: Quantity) -> Quantity:
        """
        dN/dlogM = M * dN/dM.
        """
        M = M.to(self.Mmin.unit)
        return M * self.dNdM(M)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def get_normalization(self, Mtot: Quantity) -> Quantity:
        """
        Solve for the normalization constant A given total stellar mass Mtot.

        Parameters
        ----------
        Mtot : Quantity
            Total stellar mass to normalize the IMF to.

        Returns
        -------
        A : Quantity
            Normalization constant (units 1/Msun) such that

                ∫_{Mmin}^{Mmax} [ M * dN/dM ] dM = Mtot.
        """
        beta = self.beta
        expo = 2.0 - beta  # exponent for ∫ M^{1-α} dM

        if np.isclose(expo, 0.0):
            # special case: 2 - beta = 0 → integral is log(M)
            integral = np.log(self.Mmax / self.Mmin)
        else:
            integral = (self.Mmax ** expo - self.Mmin ** expo) / expo

        # dN/dM has units 1/Msun, so A must as well
        self._A = (Mtot / integral).to(1 / u.Msun)
        return self._A

    # ------------------------------------------------------------------
    # Fractions (independent of overall normalization)
    # ------------------------------------------------------------------
    def number_fraction(self, M1: Quantity, M2: Quantity) -> float:
        """
        Fraction of *stars by number* in [M1, M2] relative to the total IMF.

        Uses dN/dM ∝ M^{-α}. The overall normalization cancels.

        Returns
        -------
        float
            Dimensionless fraction in [0, 1].
        """
        beta = self.beta
        M1, M2 = M1.to(self.Mmin.unit), M2.to(self.Mmin.unit)

        # order & clip
        lo = np.maximum(np.minimum(M1, M2), self.Mmin)
        hi = np.minimum(np.maximum(M1, M2), self.Mmax)
        if hi <= lo:
            return 0.0

        e_num = 1.0 - beta  # exponent for ∫ M^{-α} dM

        def I_num(a: Quantity, b: Quantity) -> Quantity:
            if np.isclose(e_num, 0.0, atol=1e-12):
                # beta = 1 → log-flat IMF
                return np.log(b / a)
            return (b ** e_num - a ** e_num) / e_num

        N_range = I_num(lo, hi)
        N_tot = I_num(self.Mmin, self.Mmax)
        return (N_range / N_tot).to_value(u.one)

    def mass_fraction(self, M1: Quantity, M2: Quantity) -> float:
        """
        Fraction of *total stellar mass* in [M1, M2] relative to the total IMF mass.

        Uses M * dN/dM ∝ M^{1-α}. The normalization cancels.

        Returns
        -------
        float
            Dimensionless fraction in [0, 1].
        """
        beta = self.beta
        M1, M2 = M1.to(self.Mmin.unit), M2.to(self.Mmin.unit)

        # order & clip
        lo = np.maximum(np.minimum(M1, M2), self.Mmin)
        hi = np.minimum(np.maximum(M1, M2), self.Mmax)
        if hi <= lo:
            return 0.0

        e_mass = 2.0 - beta  # exponent for ∫ M^{1-α} dM

        def I_mass(a: Quantity, b: Quantity) -> Quantity:
            if np.isclose(e_mass, 0.0, atol=1e-12):
                # beta = 2 → mass integral becomes log
                return np.log(b / a)
            return (b ** e_mass - a ** e_mass) / e_mass

        M_range = I_mass(lo, hi)
        M_tot = I_mass(self.Mmin, self.Mmax)
        return (M_range / M_tot).to_value(u.one)
