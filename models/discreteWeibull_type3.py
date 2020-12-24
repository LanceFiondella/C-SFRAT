import math
import numpy as np

import symengine

from core.model import Model


class DiscreteWeibullType3(Model):
    name = "Discrete Weibull (Type III)"
    shortName = "DW-III"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    # b0 = 0.01
    beta0 = 0.01

    parameterEstimates = (0.1, 0.5)

    def hazardFunction(self, i, args):
        # c, beta
        f = 1 - math.exp(-args[0] * i**args[1])
        return f

    def hazard_symbolic(self, i, args):
        # c, beta
        f = 1 - symengine.exp(-args[0] * i**args[1])
        return f
