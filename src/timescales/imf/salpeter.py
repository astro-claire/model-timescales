# src/timescales/profiles/power_law_BH.py
from __future__ import annotations
import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy import constants as const

from .base import imfBase
from ..factoryimf import register_imf
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