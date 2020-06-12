#import logging
#import numpy as np

from core.model import Model


class Geometric(Model):
    name = "Geometric"
    shortName = "GM"
    coxParameterEstimateRange = [0.0, 0.1]      # betas
    shapeParameterEstimateRange = [0.8, 0.99]   # b0
    # LLFspecified = False
    # dLLFspecified = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        LLF_array = [self.LLF0, self.LLF1, self.LLF2, self.LLF3]
        dLLF_array = [self.dLLF0, self.dLLF1, self.dLLF2, self.dLLF3]

    # def calcHazard(self, b, n):
    #     # return [b for i in range(self.n)]
    #     return [b for i in range(n)]

    def hazardFunction(self, i, b):
        # b = symbols("b")
        f = b
        return f

    def LLF0(self):
        pass

    def LLF1(self):
        pass

    def LLF2(self):
        pass

    def LLF3(self):
        pass

    def dLLF0(self):
        pass

    def dLLF1(self):
        pass

    def dLLF2(self):
        pass

    def dLLF3(self):
        pass

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