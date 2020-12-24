import math
import numpy as np

import symengine

from core.model import Model


class DFR_SB(Model):
    name = "DFR Salvia & Bollinger"
    shortName = "DFR SB"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    # b0 = 0.01
    beta0 = 0.01

    parameterEstimates = (0.1, )

    def hazardFunction(self, i, args):
        # c
        f = args[0] / i
        return f

    def hazard_symbolic(self, i, args):
        # c
        f = args[0] / i
        return f
