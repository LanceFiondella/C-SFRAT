from core.model import Model


class NegativeBinomial2(Model):
    name = "Negative Binomial (Order 2)"
    shortName = "NB2"

    # initial parameter estimates
    beta0 = 0.01
    parameterEstimates = (0.01,)

    def hazardSymbolic(self, i, args):
        f = (i * args[0]**2)/(1 + args[0] * (i - 1))
        return f

    def hazardNumerical(self, i, args):
        f = (i * args[0]**2)/(1 + args[0] * (i - 1))
        return f
