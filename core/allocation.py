from scipy.optimize import shgo
import numpy as np


class EffortAllocation:
    def __init__(self, model, B, f, covariate_data):
        self.model = model
        self.B = B
        self.f = f
        self.covariate_data = covariate_data
        self.hazard_array = np.concatenate((self.model.hazard_array, [self.model.hazardFunction(self.model.n + 1, self.model.b)]))
        self.runAllocation()
        self.runAllocation2()
        self.percentages = self.organizeResults(self.res.x, self.B)
        self.percentages2 = self.organizeResults(self.res2.x, self.budget)

    def runAllocation(self):

        ##############################################
        ## Optimization 1: Maximize fault discovery ##
        ##      optimal allocation of budget B      ##
        ##############################################

        cons = ({'type': 'ineq', 'fun': lambda x: self.B - sum([x[i] for i in range(self.model.numCovariates)])})
        bnds = tuple((0, None) for i in range(self.model.numCovariates))

        self.res = shgo(self.allocationFunction, args=(self.covariate_data,), bounds=bnds, constraints=cons)#, n=10000, iters=4)
        # print(self.res)
        self.mvfVal = -self.res.fun
        self.H = self.mvfVal - self.model.mvfList[-1]   # predicted MVF value - last actual MVF value

    def allocationFunction(self, x, covariate_data):
        # covariate_data = args[1]
        new_cov_data = np.concatenate((covariate_data, x[:, None]), axis=1)
        omega = self.model.calcOmega(self.hazard_array, self.model.betas, new_cov_data)

        # must be negative, SHGO uses minimization and we want to maximize fault discovery
        return -(self.model.MVF(self.model.mle_array, omega, self.hazard_array, new_cov_data.shape[1] - 1, new_cov_data))

    def runAllocation2(self):
        #####################################
        ## Optimization 2: Minimize budget ##
        ##  identify m additional faults   ##
        #####################################

        cons2 = ({'type': 'eq', 'fun': self.optimization2, 'args': (self.covariate_data,)})
        bnds = tuple((0, None) for i in range(self.model.numCovariates))

        self.res2 = shgo(lambda x: sum([x[i] for i in range(self.model.numCovariates)]), bounds=bnds, constraints=cons2)
        self.effort = np.sum(self.res2.x)


        ## allocation type 2 not integrated into UI yet
        ## just print results for now
        # print(self.res2)
        # print(self.allocationFunction2(self.res2.x, self.covariate_data))
        # print(np.multiply(np.divide(self.res2.x, np.sum(self.res2.x)), 100))


        print(f'{self.model.name} - ({self.model.metricString})')
        print("Effort per covariate:", self.res2.x)
        print("Total effort:", self.effort)
        print("Effort percentages:", np.multiply(np.divide(self.res2.x, np.sum(self.res2.x)), 100))
        print()



    def optimization2(self, x, covariate_data):
        res = self.allocationFunction2(x, covariate_data)
        H = res - self.model.mvfList[-1]
        return self.f - H
        # return self.model.allocationFunction2(x) - self.f

    def allocationFunction2(self, x, covariate_data):
        # failures = args[0]
        # i = self.n + failures
        new_cov_data = np.concatenate((covariate_data, x[:, None]), axis=1)
        omega = self.model.calcOmega(self.hazard_array, self.model.betas, new_cov_data)

        # we want to minimize, SHGO uses minimization
        return self.model.MVF(self.model.mle_array, omega, self.hazard_array, new_cov_data.shape[1] - 1, new_cov_data)

    def organizeResults(self, results, budget):
        self.percentages = np.multiply(np.divide(results, budget), 100)
