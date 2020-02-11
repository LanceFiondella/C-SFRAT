from scipy.optimize import shgo
import numpy as np

# def effortAllocation(model, B, f):
#     # cons = ({'type': 'ineq', 'fun': lambda x:  B-x[0]-x[1]-x[2]})
#     cons = ({'type': 'ineq', 'fun': lambda x:  B - sum([x[i] for i in range(model.numCovariates)])})
#     # bnds = ((0, None), (0, None), (0, None))
#     bnds = tuple((0, None) for i in range(model.numCovariates))

#     res = shgo(model.allocationFunction, args=(f,), bounds=bnds, constraints=cons)#, n=10000, iters=4)
#     # res = shgo(lambda x: -(51+ 1.5449911694401008*(1- (0.9441308828628996 ** (np.exp(0.10847739229960603*x[0]+0.027716725008716442*x[1]+0.159319065848297*x[2]))))), bounds=bnds, constraints=cons, n=10000, iters=4)
#     print(res)
#     print(sum(res.x))

#     return res

class EffortAllocation:
    def __init__(self, model, B, f):
        self.model = model
        self.B = B
        self.f = f
        self.runAllocation()
        self.organizeResults()
    
    def runAllocation(self):
        # cons = ({'type': 'ineq', 'fun': lambda x:  B-x[0]-x[1]-x[2]})
        cons = ({'type': 'ineq', 'fun': lambda x:  self.B - sum([x[i] for i in range(self.model.numCovariates)])})
        # bnds = ((0, None), (0, None), (0, None))
        bnds = tuple((0, None) for i in range(self.model.numCovariates))

        self.res = shgo(self.model.allocationFunction, args=(self.f,), bounds=bnds, constraints=cons)#, n=10000, iters=4)
        self.mvfVal = -self.res.fun
        self.H = self.mvfVal - self.model.mvfList[-1]   # predicted MVF value - last actual MVF value
        # self.H = self.model.MVF(self.model.calcHazard(self.model.b), self.model.omega, self.model.betas, self.model.n + 1) - self.model.mvfList[-1]
            # may need to change from calcHazard function

    def organizeResults(self):
        # self.H = self.res.funl[0]   # function evaluation
        self.percentages = np.multiply(np.divide(self.res.x, self.B), 100)