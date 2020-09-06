from scipy.optimize import shgo
import numpy as np


class EffortAllocation:
    def __init__(self, model, B, f):
        self.model = model
        self.B = B
        self.f = f
        self.runAllocation()
        self.organizeResults()

    def runAllocation(self):

        ##############################################
        ## Optimization 1: Maximize fault discovery ##
        ##      optimal allocation of budget B      ##
        ##############################################

        # cons = ({'type': 'ineq', 'fun': lambda x:  B-x[0]-x[1]-x[2]})
        cons = ({'type': 'ineq', 'fun': lambda x: self.B - sum([x[i] for i in range(self.model.numCovariates)])})
        # bnds = ((0, None), (0, None), (0, None))
        bnds = tuple((0, None) for i in range(self.model.numCovariates))

        self.res = shgo(self.model.allocationFunction, args=(self.f,), bounds=bnds, constraints=cons)#, n=10000, iters=4)
        self.mvfVal = -self.res.fun
        self.H = self.mvfVal - self.model.mvfList[-1]   # predicted MVF value - last actual MVF value

        # hOmegaN plus one
        # print(self.res)

        # self.H = self.model.MVF(self.model.calcHazard(self.model.b), self.model.omega, self.model.betas, self.model.n + 1) - self.model.mvfList[-1]
            # may need to change from calcHazard function

        #####################################
        ## Optimization 2: Minimize budget ##
        ##  identify m additional faults   ##
        #####################################

        # cons2 = ({'type': 'ineq', 'fun': lambda x: self.B - sum([x[i] for i in range(self.model.numCovariates)])},
        #         {'type': 'ineq', 'fun': self.optimization2})
        cons2 = ({'type': 'eq', 'fun': self.optimization2})

        # cons2 = ({'type': 'ineq', 'fun': lambda x: self.f - self.model.allocationFunction2(x)})
        # self.res2 = shgo(self.model.allocationFunction2, args=(self.f,), bounds=bnds, constraints=cons2)
        self.res2 = shgo(lambda x: sum([x[i] for i in range(self.model.numCovariates)]), bounds=bnds, constraints=cons2)
        print(self.res2)
        print(self.model.allocationFunction2(self.res2.x))
        print(np.multiply(np.divide(self.res2.x, np.sum(self.res2.x)), 100))






    def optimization2(self, x):
        res = self.model.allocationFunction2(x)
        H = res - self.model.mvfList[-1]
        return self.f - H
        # return self.model.allocationFunction2(x) - self.f

    def organizeResults(self):
        # self.H = self.res.funl[0]   # function evaluation
        self.percentages = np.multiply(np.divide(self.res.x, self.B), 100)
