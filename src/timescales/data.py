from .utils import kinetic_energy, gravitational_potential_energy, as_quantity
import numpy as np
import astropy.units as u
from astropy.units import Quantity
import astropy.constants as c
import scipy.special 

def build_bulk_energy_grid(masses: Quantity,
                           radii: Quantity,
                           velocities: Quantity,
                           *,
                           alpha=3/5,
                           energy_unit=u.erg,
                           ratio_max=15,           # if set (e.g., 10), keep only (-U/K) <= ratio_max
                           bound_only=True,          # if True, drop systems with K+U >= 0
                           return_mask=False):
    """
    Create aligned 1D arrays for all (M, R, V) combos and their energies.
    Optionally:
      - keep only bound systems (K+U < 0),
      - keep only systems with (-U/K) <= ratio_max.

    Returns dict with keys: 'M','R','V','K','U' (Quantities).
    If `return_mask=True`, also includes 'mask' (np.bool_) and 'minusU_over_K' (dimensionless).
    """
    # Mesh the 3 axes (shape Nm x Nr x Nv)
    M3, R3, V3 = np.meshgrid(masses, radii, velocities, indexing='ij')

    # Energies
    K3 = kinetic_energy(M3, V3, out_unit=energy_unit)
    U3 = gravitational_potential_energy(M3, R3, alpha=alpha, out_unit=energy_unit)

    # Flatten
    M = M3.ravel(); R = R3.ravel(); V = V3.ravel()
    K = K3.ravel(); U = U3.ravel()

    # Boundness: E = K + U < 0  (equivalently, -U/K > 1 when K>0)
    E = (K + U)  # Quantity with energy units
    mask = np.ones(E.shape, dtype=bool)
    if bound_only:
        mask &= (E < 0 * energy_unit)

    # Ratio r = -U/K (dimensionless). We compute this after bound mask so most NaNs/Infs are already gone.
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (-U / K).to_value(u.one)

    # Apply threshold if provided: keep where (-U/K) <= ratio_max
    if ratio_max is not None:
        # When applying the threshold, require r to be finite
        mask &= np.isfinite(r) & (r <= ratio_max)

    # Filter all arrays
    out = {
        'M': M[mask],
        'R': R[mask],
        'V': V[mask],
        'K': K[mask],
        'U': U[mask],
    }
    if return_mask:
        out['mask'] = mask
        out['minusU_over_K'] = (r[mask]) * u.one

    return out