from core.model import Model


class IFR_Generalized_SB(Model):
    name = "IFR generalized Salvia & Bollinger"
    shortName = "IFRGSB"

    # initial parameter estimates
    beta0 = 0.01
    parameterEstimates = (0.1, 0.1)

    def hazardSymbolic(self, i, args):
        # args -> (c, alpha)
        f = 1 - args[0] / ((i - 1) * args[1] + 1)
        return f

    def hazardNumerical(self, i, args):
        # args -> (c, alpha)
        f = 1 - args[0] / ((i - 1) * args[1] + 1)
        return f
