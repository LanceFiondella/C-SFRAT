"""
import math
import numpy as np

import symengine

from core.model import Model


class DiscreteWeibullType1(Model):
    name = "Discrete Weibull (Type I)"
    shortName = "DW-I"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    # b0 = 0.01
    beta0 = 0.1

    parameterEstimates = (0.1, 0.1)

    def hazardFunction(self, i, args):
        # q, beta
        f = 1 - args[0]**(i**args[1] - (i - 1)**args[1])
        return f

    def hazard_symbolic(self, i, args):
        # q, beta
        f = 1 - args[0]**(i**args[1] - (i - 1)**args[1])
        return f
"""
