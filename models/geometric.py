#import logging
#import numpy as np

from core.model import Model


class Geometric(Model):
    name = "Geometric"
    shortName = "GM"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def hazardFunction(self, i, b):
        f = b
        return f
