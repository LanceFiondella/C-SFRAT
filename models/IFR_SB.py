import math
import numpy as np

import symengine

from core.model import Model


class IFR_SB(Model):
    name = "IFR Salvia & Bollinger"
    shortName = "IFR SB"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    # b0 = 0.01
    beta0 = 0.01

    parameterEstimates = (0.1, )

    def hazardFunction(self, i, args):
        # alpha, beta
        f = 1 - args[0] / i
        return f

    def hazard_symbolic(self, i, args):
        f = 1 - args[0] / i
        return f
