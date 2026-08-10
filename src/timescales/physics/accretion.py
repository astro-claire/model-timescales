import numpy as np
import astropy.units as u
import astropy.constants as c



def stellar_bondi_accretion_rate(rhogas, Mstar, cs, v):
    return (4 * np.pi * c.G**2 * Mstar**2 / (cs**2+ v**2)**(3./2)).to(u.Msun/u.yr)


def smbh_bondi_accretion_rate(rhogas, MBH, cs):
    return (4 * np.pi * c.G**2 * MBH**2 / (cs**3)).to(u.Msun/u.yr)


def bondi_radius(M,cs):
    return (c.G * M/ (cs**2)).to('pc')