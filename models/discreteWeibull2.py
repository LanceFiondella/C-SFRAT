import logging
import numpy as np

from core.model import Model


class DiscreteWeibull2(Model):
    name = "Discrete Weibull (Order 2)"
    shortName = "DW2"
    coxParameterEstimateRange = [0.0001, 0.01]      # betas
    shapeParameterEstimateRange = [0.9, 0.9999]   # b0

    b0 = 0.994
    beta0 = 0.01

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def hazardFunction(self, i, b):
        f = 1 - b**(i**2 - (i - 1)**2)
        return f