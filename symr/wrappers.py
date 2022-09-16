import os

import numpy as np

from sympy import lambdify
import sympy as sp

import abc
from abc import abstractmethod


class BaseWrapper(metaclass=abc.ABCMeta):
    """
    abstract base class for SR method wrappers
    these methods are never called, but they must be overriden by inheriting classes
    therefore they are excluded from coverage. 
    """
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, target, **kwargs): #pragma: no cover 
        """
        target is e.g. the y value or other quantity returned by the function
        **kwargs contains the input variables, which may include several input variables
        """
        pass

    @abstractmethod
    def initialize_model(self): #pragma: no cover 
        pass

    @abstractmethod
    def load_parameters(self): #pragma: no cover 
        pass

