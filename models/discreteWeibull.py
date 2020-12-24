"""
import math
import numpy as np

import symengine

from core.model import Model


class DiscreteWeibull(Model):
    name = "Discrete Weibull"
    shortName = "DW"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    # b0 = 0.01
    beta0 = 0.01

    parameterEstimates = (0.1, 0.1)

    def hazardFunction(self, i, args):
        # alpha, beta
        f = 1 - math.exp(((i - 1)**args[1] - i**args[1]) / args[0])
        # f = 1 - symengine.exp(args[0])
        return f

    def hazard_symbolic(self, i, args):
        f = 1 - symengine.exp(((i - 1)**args[1] - i**args[1]) / args[0])
        # f = 1 - symengine.exp(args[0])
        return f
"""
