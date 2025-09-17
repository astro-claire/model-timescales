#src/timescales/imf/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional


class imfBase(ABC):
    """ 
    Abstract base class for initial mass functions
    """
    def __init(self,**kwargs):
        self._init_kwargs: Dict[str, Any] = dict(kwargs)

    @property
    def params(self) -> Mapping[str, Any]:
        """
        Return a read-only mapping of model parameters for logging/serialization.
        Subclasses may override to expose a curated dict.
        """
        return dict(self._init_kwargs)

    #--------required interface----------
    @abstractmethod
    def dNdlogM(self):
        """ 
        initial mass function: 
            dN/dlogM
        """
        raise NotImplementedError

    def dNdM(self):
        """ 
        Mass spectrum:
            dN/dM (integrate to get total mass)
        """
        raise NotImplementedError

    def number_fraction(self):
        raise NotImplementedError