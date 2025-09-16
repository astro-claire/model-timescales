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
    def __init__(self, k1 = 1, k2 = 1):
        self.k1 = k1
        self.k2 = k2

    def dNdlogM(self, M):
        return self.k1 * M**(-1.35)
    
    def dNdM(self, M):
        return self.k2 * M **(-2.35)

    def get_normalization(self, Mtot):
        pass