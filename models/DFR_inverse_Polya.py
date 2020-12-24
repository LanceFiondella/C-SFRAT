import math
import numpy as np

import symengine

from core.model import Model


class DFR_Inverse_Polya(Model):
    name = "DFR inverse Polya"
    shortName = "DFR IP"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    # b0 = 0.01
    beta0 = 0.01

    parameterEstimates = (0.1, 0.1, 0.1)

    def hazardFunction(self, i, args):
        # r, w, delta
        f = args[0] / (args[0] + args[1] + (i - 1) * args[2])
        return f

    def hazard_symbolic(self, i, args):
        # p, pi
        f = args[0] / (args[0] + args[1] + (i - 1) * args[2])
        return f
