import math
import symengine

from core.model import Model


class TruncatedLogistic(Model):
    name = "Truncated Logistic"
    shortName = "TL"

    # initial parameter estimates
    beta0 = 0.01
    parameterEstimates = (0.1, 0.1)

    def hazardSymbolic(self, i, args):
        # args -> (c, d)
        f = (1 - symengine.exp(-1/args[1]))/(1 + symengine.exp(- (i - args[0])/args[1]))
        return f

    def hazardNumerical(self, i, args):
        # args -> (c, d)
        f = (1 - math.exp(-1/args[1]))/(1 + math.exp(- (i - args[0])/args[1]))
        return f
