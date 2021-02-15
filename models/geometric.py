from core.model import Model


class Geometric(Model):
    name = "Geometric"
    shortName = "GM"

    # initial parameter estimates
    beta0 = 0.01
    parameterEstimates = (0.01,)

    def hazardSymbolic(self, i, args):
        f = args[0]
        return f

    def hazardNumerical(self, i, args):
        f = args[0]
        return f
