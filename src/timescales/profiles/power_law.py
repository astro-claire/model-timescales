# src/timescales/profiles/power_law.py
from __future__ import annotations
import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy import constants as const

from .base import ProfileBase
from ..factory import register_profile
from ..utils.units import as_quantity  # your helper for coercion

@register_profile("power-law", aliases=("pl"))
class PowerLawProfile(ProfileBase):
    """
    Spherical power-law structure:
        ρ(r) = ρ0 * (r / r0)^(-α),  with optional inner cutoff r_min > 0 if α ≥ 3.
    """

    def __init__(
        self,
        *,
        alpha: float,
        r0: Quantity,
        rho0: Quantity | None = None,
        # Alternatively, normalize from a mass constraint M_ref within R_ref:
        M_ref: Quantity | None = None,
        R_ref: Quantity | None = None,
        r_min: Quantity = 0 * u.cm,
    ) -> None:
        super().__init__(
            alpha=alpha, r0=r0, rho0=rho0, M_ref=M_ref, R_ref=R_ref, r_min=r_min
        )
        self.alpha = float(alpha)
        if r0 is None:
            r0 = R_ref
        self.r0 = as_quantity(r0, u.pc)
        self.r_min = as_quantity(r_min, self.r0.unit)

        # Determine normalization ρ0 either directly or from mass constraint
        if rho0 is not None:
            self.rho0 = as_quantity(rho0, u.Msun / u.pc**3)
        else:
            if (M_ref is None) or (R_ref is None):
                raise ValueError("Provide rho0 directly OR (M_ref, R_ref) for normalization.")
            self.rho0 = self._rho0_from_mass_constraint(
                M=as_quantity(M_ref, u.Msun),
                R=as_quantity(R_ref, self.r0.unit),
            )

        # Guard against divergence
        if self.alpha >= 3.0 and np.all(self.r_min == 0 * self.r0.unit):
            raise ValueError("For alpha ≥ 3, r_min must be > 0 to avoid divergence.")

    # -------------------------- required interface --------------------------

    def density(self, r: Quantity) -> Quantity:
        r = as_quantity(r, self.r0.unit)
        # ρ(r) = ρ0 (r/r0)^(-α); treated piecewise if r <= r_min (you may choose 0 or clip)
        rho = self.rho0 * (r / self.r0) ** (-self.alpha)
        # Optionally: zero (or NaN) inside r_min if you enforce a hard inner cutoff
        mask_inner = r <= self.r_min
        if np.any(mask_inner):
            rho = rho.copy()
            rho[mask_inner] = 0 * rho.unit
        return rho

    def enclosed_mass(self, r: Quantity) -> Quantity:
        """
        M(<r) = 4π ρ0 r0^α [ (r^(3-α) - r_min^(3-α)) / (3-α) ]  (α ≠ 3)
               = 4π ρ0 r0^3 ln(r / r_min)                      (α = 3)
        """
        r = as_quantity(r, self.r0.unit)
        r_eff = np.maximum(r, self.r_min)  # avoid log of <1 if you prefer; or treat piecewise

        prefac = 4 * np.pi * self.rho0 * (self.r0 ** self.alpha)

        if np.isclose(self.alpha, 3.0, atol=1e-12):
            if np.any(self.r_min <= 0 * self.r0.unit):
                raise ValueError("α=3 requires r_min > 0.")
            M = (prefac * np.log(r_eff / self.r_min)).to(u.Msun)
        else:
            denom = (3.0 - self.alpha)
            term = (r_eff ** (3.0 - self.alpha)) - (self.r_min ** (3.0 - self.alpha))
            M = (prefac * term / denom).to(u.Msun)

        return M

    def velocity_dispersion(self, r: Quantity) -> Quantity:
        """
        σ(r): choose either a closed-form (when available) or call a generic Jeans solver.

        For an isotropic, spherical power-law under common assumptions one often uses:
            σ(r) ≈ [ G M(<r) / ((1 + α) r) ]^1/2
        (Adjust according to your exact modeling choice / reference.)
        """
        r = as_quantity(r, self.r0.unit)
        Menc = self.enclosed_mass(r)
        sigma2 = const.G * Menc / (r * (1.0 + self.alpha))
        return np.sqrt(sigma2).to(u.km / u.s)

    # ---------------------------- helpers -----------------------------------

    def _rho0_from_mass_constraint(self, M: Quantity, R: Quantity,M_unit=u.Msun, r_unit=u.pc) -> Quantity:
        """Solve for ρ0 given M(<R). Handles the α→3 limit with the log form."""
        # four_pi = 4.0 * np.pi
        # if np.isclose(self.alpha, 3.0, atol=1e-12):
        #     if np.any(self.r_min <= 0 * self.r0.unit):
        #         raise ValueError("α=3 requires r_min > 0 for normalization.")
        #     rho0 = (M_ref / (four_pi * (self.r0 ** 3) * np.log(R / self.r_min))).to(u.Msun / u.pc**3)
        # else:
        #     num = M_ref * (3.0 - self.alpha)
        #     den = four_pi * (self.r0 ** self.alpha) * (R ** (3.0 - self.alpha) - self.r_min ** (3.0 - self.alpha))
        #     rho0 = (num / den).to(u.Msun / u.pc**3)
        # return rho0

        # Broadcast everything
        alpha = np.asarray(self.alpha, dtype=float)
        M, R, r0, r_min, alpha = np.broadcast_arrays(M, R, self.r0, self.r_min, alpha)
        M = M*M_unit
        R = R* r_unit
        r0 = r0*r_unit
        r_min = r_min * r_unit

        # Guards
        if np.any(R <= r_min):
            raise ValueError("R must be strictly greater than r_min.")
        if np.any((alpha >= 3.0) & (r_min == 0 * r_unit)):
            raise ValueError("For alpha >= 3, r_min must be > 0 to avoid divergence.")

        # Precompute factors
        four_pi = 4.0 * np.pi

        # Allocate result
        rho0 = np.empty(M.shape, dtype=float) * (M_unit / r_unit**3)

        # alpha ≈ 3 branch (log form)
        mask_log = np.isclose(alpha, 3.0, atol=1e-12)
        if np.any(mask_log):
            rho0[mask_log] = (M[mask_log] / (four_pi * r0[mask_log]**3 *
                                            np.log(R[mask_log] / r_min[mask_log]))).to(rho0.unit)

        # general branch (alpha != 3)
        mask_gen = ~mask_log
        if np.any(mask_gen):
            num = M[mask_gen] * (3.0 - alpha[mask_gen])
            den = four_pi * (r0[mask_gen] ** alpha[mask_gen]) * ( R[mask_gen] ** (3.0 - alpha[mask_gen]) - r_min[mask_gen] ** (3.0 - alpha[mask_gen]))
            rho0[mask_gen] = (num / den)
        return rho0
