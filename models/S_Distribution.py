from core.model import Model


class S_Distribution(Model):
    name = "S Distribution"
    shortName = "S"

    # initial parameter estimates
    beta0 = 0.01
    parameterEstimates = (0.1, 0.1)

    def hazardSymbolic(self, i, args):
        # args -> (p, pi)
        f = args[0] * (1 - args[1]**i)
        return f

    def hazardNumerical(self, i, args):
        # args -> (p, pi)
        f = args[0] * (1 - args[1]**i)
        return f
