# src/timescales/imf/salpeter.py
from __future__ import annotations
import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy import constants as const

from .base import imfBase
from .registry import register_imf
from ..utils.units import as_quantity  # your helper for coercion


@register_imf("salpeter", aliases=("sp","saltpeter"))
class salpeterIMF(imfBase):
    """ 
    Salpeter imf with gamma = 1.35
    """
    def __init__(self, Mmin=0.1*u.Msun, Mmax=100*u.Msun):
        self.alpha = 2.35
        self.Mmin = Mmin
        self.Mmax = Mmax
        self._A = None   # normalization constant

    def dNdM(self, M: Quantity) -> Quantity:
        """
        Mass spectrum dN/dM.
        Requires normalization constant self._A to be set.
        """
        if self._A is None:
            raise RuntimeError("IMF not normalized yet. Call get_normalization first.")
        return self._A * (M**(-self.alpha))

    
    def dNdlogM(self, M: Quantity) -> Quantity:
        """
        dN/dlogM = M * dN/dM
        """
        return M * self.dNdM(M)

    def get_normalization(self, Mtot: Quantity) -> float:
        """
        Solve for the normalization constant A given total stellar mass Mtot.

        Parameters
        ----------
        Mtot : Quantity
            Total stellar mass to normalize the IMF to.

        Returns
        -------
        A : float with units 1/Msun
            Normalization constant such that ∫ M (dN/dM) dM = Mtot.
        """
        # integral of M^(1 - alpha) from Mmin to Mmax
        expo = 2 - self.alpha
        if np.isclose(expo, 0.0):
            integral = np.log(self.Mmax / self.Mmin)
        else:
            integral = (self.Mmax**expo - self.Mmin**expo) / expo

        self._A = (Mtot / integral).to(1/u.Msun)  # dN/dM has units 1/Msun
        return self._A

    def number_fraction(self, M1, M2):
        """
        Fraction of stars between M1 and M2 (by number, not mass).

        Parameters
        ----------
        M1, M2 : Quantity
            Lower and upper mass bounds.

        Returns
        -------
        frac : float
            Fraction of stars in [M1, M2] relative to the total IMF.
        """
        alpha = self.alpha
        expo = 1 - alpha  # exponent for integrating dN/dM ~ M^-alpha

        def integral(Mlo, Mhi):
            if np.isclose(expo, 0.0):
                return np.log(Mhi / Mlo)
            else:
                return (Mhi**expo - Mlo**expo) / expo

        # compute in same units
        M1 = M1.to(self.Mmin.unit)
        M2 = M2.to(self.Mmin.unit)

        N_range = integral(M1, M2)
        N_tot = integral(self.Mmin, self.Mmax)

        return (N_range / N_tot).value

    def number_fraction(self, M1: Quantity, M2: Quantity) -> float:
        """
        Fraction of *stars by number* in [M1, M2] relative to the total IMF.

        Uses dN/dM ∝ M^{-alpha}. The normalization cancels in the ratio.

        Returns
        -------
        float
            Dimensionless fraction in [0, 1].
        """
        alpha = self.alpha
        M1, M2 = M1.to(self.Mmin.unit), M2.to(self.Mmin.unit)
        # ensure ordered and clipped to support bounds
        lo = np.maximum(np.minimum(M1, M2), self.Mmin)
        hi = np.minimum(np.maximum(M1, M2), self.Mmax)
        if hi <= lo:
            return 0.0

        # ∫ M^{-alpha} dM = (M^{1-α})/(1-α), unless 1-α≈0 → log
        e_num = 1.0 - alpha

        def I_num(a: Quantity, b: Quantity) -> Quantity:
            if np.isclose(e_num, 0.0, atol=1e-12):
                return np.log(b / a)
            return (b**e_num - a**e_num) / e_num

        N_range = I_num(lo, hi)
        N_tot   = I_num(self.Mmin, self.Mmax)
        return (N_range / N_tot).to_value(u.one)
        
    def mass_fraction(self, M1: Quantity, M2: Quantity) -> float:
        """
        Fraction of *total stellar mass* in [M1, M2] relative to the total IMF mass.

        Uses M * dN/dM ∝ M^{1-α}. The normalization cancels in the ratio.

        Returns
        -------
        float
            Dimensionless fraction in [0, 1].
        """
        alpha = self.alpha
        M1, M2 = M1.to(self.Mmin.unit), M2.to(self.Mmin.unit)
        # ensure ordered and clipped to support bounds
        lo = np.maximum(np.minimum(M1, M2), self.Mmin)
        hi = np.minimum(np.maximum(M1, M2), self.Mmax)
        if hi <= lo:
            return 0.0

        # ∫ M^{1-α} dM = (M^{2-α})/(2-α), unless 2-α≈0 → log
        e_mass = 2.0 - alpha

        def I_mass(a: Quantity, b: Quantity) -> Quantity:
            if np.isclose(e_mass, 0.0, atol=1e-12):
                return np.log(b / a)
            return (b**e_mass - a**e_mass) / e_mass

        M_range = I_mass(lo, hi)
        M_tot   = I_mass(self.Mmin, self.Mmax)
        return (M_range / M_tot).to_value(u.one)