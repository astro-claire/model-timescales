"""
Model for each system! 
"""

from __future__ import annotations
import astropy.units as u
from astropy.units import Quantity
import warnings
from .tables import structural_table, timescale_table
from .tools import get_system, condition_test
from ..physics.stars import main_sequence_lifetime_approximation, stellar_radius_approximation  
from ..physics.halo_environment import local_merger_timescale, neighbor_merger_timescale, interaction_timescale
from .recipes import get
from ..utils.energy import escape_velocity