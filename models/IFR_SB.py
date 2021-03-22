from core.model import Model


class IFR_SB(Model):
    name = "IFR Salvia & Bollinger"
    shortName = "IFR SB"

    # initial parameter estimates
    beta0 = 0.01
    parameterEstimates = (0.1, )

    def hazardSymbolic(self, i, args):
        f = 1 - args[0] / i
        return f

    def hazardNumerical(self, i, args):
        # alpha, beta
        f = 1 - args[0] / i
        return f
