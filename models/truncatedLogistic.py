import math
import numpy as np

import symengine

from core.model import Model


class TruncatedLogistic(Model):
    name = "Truncated Logistic"
    shortName = "TL"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    # b0 = 0.01
    # beta0 = 0.498569
    beta0 = 0.01

    # parameterEstimates = (-331.0, 10.17)
    parameterEstimates = (0.1, 0.1)

    def hazardFunction(self, i, args):
        # c, d
        # f = (math.exp(-1 / args[1]) * (-1 + math.exp(1 / args[1]))) / (1 + math.exp((args[0] - i) / args[1]))
        f = (1 - math.exp(-1/args[1]))/(1 + math.exp(- (i - args[0])/args[1]))
        # f = (math.exp(args[0] / args[1])) / (1 + math.exp((args[0] - i) / args[1]))
        return f

    def hazard_symbolic(self, i, args):
        # c, d
        # f = (symengine.exp(-1 / args[1]) * (-1 + symengine.exp(1 / args[1]))) / (1 + symengine.exp((args[0] - i) / args[1]))
        f = (1 - symengine.exp(-1/args[1]))/(1 + symengine.exp(- (i - args[0])/args[1]))
        return f
