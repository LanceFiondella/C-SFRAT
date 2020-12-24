from core.model import Model


class NegativeBinomial2(Model):
    name = "Negative Binomial (Order 2)"
    shortName = "NB2"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    # b0 = 0.01
    beta0 = 0.01

    parameterEstimates = (0.01,)

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def hazardFunction(self, i, args):
        # args is tuple of hazard function parameters
        f = (i * args[0]**2)/(1 + args[0] * (i - 1))
        return f

    def hazard_symbolic(self, i, args):
        # args is tuple of hazard function parameters
        f = (i * args[0]**2)/(1 + args[0] * (i - 1))
        return f
