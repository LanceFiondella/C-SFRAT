import models
from core.model import Model


# we have model list, from importing models

def symAll(self):
    Model.maxCovariates = self.numCovariates
    f, x = self.LLF_sym(self.hazardFunction)    # pass hazard rate function
    bh = np.array([diff(f, x[i]) for i in range(self.numCovariates + 1)])
    Model.lambdaFunctionAll = self.convertSym(x, bh, "numpy")