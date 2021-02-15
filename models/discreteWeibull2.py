from core.model import Model


class DiscreteWeibull2(Model):
    name = "Discrete Weibull (Order 2)"
    shortName = "DW2"

    # initial parameter estimates
    beta0 = 0.01
    parameterEstimates = (0.994,)

    def hazardSymbolic(self, i, args):
        f = 1 - args[0]**(i**2 - (i - 1)**2)
        return f

    def hazardNumerical(self, i, args):
        f = 1 - args[0]**(i**2 - (i - 1)**2)
        return f
