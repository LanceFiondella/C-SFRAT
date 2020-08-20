import logging
import numpy as np

from core.model import Model


class NegativeBinomial2(Model):
    name = "Negative Binomial (Order 2)"
    shortName = "NB2"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    b0 = 0.01
    beta0 = 0.01

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def hazardFunction(self, i, b):
        f = (i * b**2)/(1 + b * (i - 1))
        return f