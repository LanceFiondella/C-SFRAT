import math
import numpy as np

import symengine

from core.model import Model


class DFR_Generalized_SB(Model):
    name = "DFR generalized Salvia & Bollinger"
    shortName = "DFR Gen SB"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    # b0 = 0.01
    beta0 = 0.01

    parameterEstimates = (0.1, 0.1)

    def hazardFunction(self, i, args):
        # c, alpha
        f = args[0] / ((i - 1) * args[1] + 1)
        return f

    def hazard_symbolic(self, i, args):
        # c, alpha
        f = args[0] / ((i - 1) * args[1] + 1)
        return f
