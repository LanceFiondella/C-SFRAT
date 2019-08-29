from core.model import Model
import logging
import numpy as np
from sympy import diff

class Geometric(Model):

    name = "Geometric"
    converged = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def calcHazard(self, b):
        return [b for i in range(self.n)]

    def runEstimation(self):
        initial = self.initialEstimates()
        logging.info("Initial estimates: {0}".format(initial))
        f, x = self.LLF_sym()
        bh = np.array([diff(f, x[i]) for i in range(self.numCovariates + 1)])
        logging.info("Log-likelihood differentiated.")
        logging.info("Converting symbolic equation to numpy...")
        fd = self.convertSym(x, bh, "numpy")
        logging.info("Symbolic equation converted.")
        sol = self.optimizeSolution(fd, initial)
        logging.info("Optimized solution: {0}".format(sol))

        b = sol[0]
        betas = sol[1:]
        hazard = self.calcHazard(b)
        self.modelFitting(hazard, betas)

if __name__ == "__main__":
    g = Geometric()