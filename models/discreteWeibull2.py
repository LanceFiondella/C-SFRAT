from core.model import Model


class DiscreteWeibull2(Model):
    name = "Discrete Weibull (Order 2)"
    shortName = "DW2"
    coxParameterEstimateRange = [0.0001, 0.01]      # betas
    shapeParameterEstimateRange = [0.9, 0.9999]   # b0

    # b0 = 0.994
    beta0 = 0.01

    parameterEstimates = (0.994,)

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def hazardFunction(self, i, args):
        f = 1 - args[0]**(i**2 - (i - 1)**2)
        return f
