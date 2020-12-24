import math
import numpy as np

import symengine

from core.model import Model


class S_Distribution(Model):
    name = "S Distribution"
    shortName = "S"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    # b0 = 0.01
    beta0 = 0.01

    parameterEstimates = (0.1, 0.1)

    def hazardFunction(self, i, args):
        # p, pi
        f = args[0] * (1 - args[1]**i)
        return f

    def hazard_symbolic(self, i, args):
        # p, pi
        f = args[0] * (1 - args[1]**i)
        return f
