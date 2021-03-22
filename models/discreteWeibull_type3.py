import math
import symengine

from core.model import Model


class DiscreteWeibullType3(Model):
    name = "Discrete Weibull (Type III)"
    shortName = "DW3"

    # initial parameter estimates
    beta0 = 0.01
    parameterEstimates = (0.1, 0.5)

    def hazardSymbolic(self, i, args):
        # args -> (c, b)
        f = 1 - symengine.exp(-args[0] * i**args[1])
        return f

    def hazardNumerical(self, i, args):
        # args -> (c, b)
        f = 1 - math.exp(-args[0] * i**args[1])
        return f
