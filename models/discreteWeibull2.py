import logging
import numpy as np

from core.model import Model

class DiscreteWeibull2(Model):
    name = "Discrete Weibull (Order 2)"
    coxParameterEstimateRange = [0.0, 0.01]
    shapeParameterEstimateRange = [0.998, 0.9999]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calcHazard(self, b):
        return [1 - np.power(b, (np.square(i) - np.square(i - 1))) for i in range(1, self.n+1)]

    def hazardFunction(self, i, b):
        # b = symbols("b")   
        f = 1 - b**(i**2 - (i - 1)**2)
        return f

    """
    def runEstimation(self):
        print("-------- DISCRETE WEIBULL (ORDER 2) --------")
        initial = self.initialEstimates(0.998, 0.9999)
        logging.info("Initial estimates: {0}".format(initial))
        f, x = self.LLF_sym(self.hazardFunction)    # pass hazard rate function
        bh = np.array([diff(f, x[i]) for i in range(self.numCovariates + 1)])
        logging.info("Log-likelihood differentiated.")
        logging.info("Converting symbolic equation to numpy...")
        fd = self.convertSym(x, bh, "numpy")
        logging.info("Symbolic equation converted.")
        sol = self.optimizeSolution(fd, initial)
        # sol = self.optimizeSolution(fd, [0.997816, 0.0361963, 0.0713079, 0.0584351])
        logging.info("Optimized solution: {0}".format(sol))

        self.b = sol[0]
        self.betas = sol[1:]
        hazard = self.calcHazard(self.b)
        self.modelFitting(hazard, self.betas)

        logging.info("Omega: {0}".format(self.omega))
        logging.info("Betas: {0}".format(self.betas))
        logging.info("b: {0}".format(self.b))
    """