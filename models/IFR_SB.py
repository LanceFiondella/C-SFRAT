from core.model import Model


class IFR_SB(Model):
    name = "IFR SB"
    shortName = "IFR SB"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    b0 = 0.9
    beta0 = 0.1

    # c and beta
    parameterEstimates = (0.5, 0.01)

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def hazardFunction(self, i, *args):
        f = 1 - args[0] / i
        return f