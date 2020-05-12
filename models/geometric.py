#import logging
#import numpy as np

from core.model import Model

class Geometric(Model):
    name = "Geometric"
    coxParameterEstimateRange = [0.0, 0.01]
    shapeParameterEstimateRange = [0.09, 0.1]
    symbolicDifferentiation = True

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def calcHazard(self, b):
        return [b for i in range(self.n)]

    def hazardFunction(self, i, b):
        # b = symbols("b")
        f = b
        return f

    # def LLF(self):
    #     pass

    # def dLLF(self):
    #     pass

    """
    def runEstimation(self):
        print("-------- GEOMETRIC --------")
        initial = self.initialEstimates(0.09, 0.1)
        logging.info("Initial estimates: {0}".format(initial))
        f, x = self.LLF_sym(self.hazardFunction)    # pass hazard rate function
        bh = np.array([diff(f, x[i]) for i in range(self.numCovariates + 1)])
        logging.info("Log-likelihood differentiated.")
        logging.info("Converting symbolic equation to numpy...")
        fd = self.convertSym(x, bh, "numpy")
        logging.info("Symbolic equation converted.")
        sol = self.optimizeSolution(fd, initial)
        logging.info("Optimized solution: {0}".format(sol))

        self.b = sol[0]
        self.betas = sol[1:]
        hazard = self.calcHazard(self.b)
        self.modelFitting(hazard, self.betas)

        logging.info("Omega: {0}".format(self.omega))
        logging.info("Betas: {0}".format(self.betas))
        logging.info("b: {0}".format(self.b))
    """