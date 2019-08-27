from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import sympy as sym
from sympy import symbols, diff, exp, lambdify, DeferredVector, factorial, Symbol, Idx, IndexedBase
import scipy.optimize

class Model(ABC):
    def __init__(self, *args, **kwargs):
        '''
        Initialize Model

        Keyword Args:
            data: Pandas dataframe with all required columns
        '''
        self.data = kwargs["data"]

    ##############################################
    #Properties/Members all models must implement#
    ##############################################
    @property
    @abstractmethod
    def name(self):
        '''
        Name of model (string)
        '''
        return "Generic Model"

    @property
    @abstractmethod
    def converged(self):
        '''
        Indicates whether model has converged (Bool)
        Must be set after parameters are calculated
        '''
        return True

    ################################################
    #Methods that must be implemented by all models#
    ################################################
    @abstractmethod
    def calcHazard(self):
        pass

    def initialEstimates(self):
        pass

    def LLF_sym(self, n, numCovariates, covariateData, kVec):
        #Equation (30)
        x = DeferredVector('x')
        second = []
        prodlist = []
        for i in range(n):
            sum1=1
            sum2=1
            TempTerm1 = 1
            for j in range(1, numCovariates + 1):
                    TempTerm1 = TempTerm1 * exp(covariateData[j - 1][i] * x[j])
            #print('Test: ', TempTerm1)
            sum1=1-((1-x[0]) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(1, numCovariates + 1):
                        TempTerm2 = TempTerm2 * exp(covariateData[j - 1][k] * x[j])
                #print ('Test:', TempTerm2)
                sum2 = sum2*((1-x[0])**(TempTerm2))
            #print ('Sum2:', sum2)
            second.append(sum2)
            prodlist.append(sum1*sum2)

        firstTerm = -sum(kVec) #Verified
        secondTerm = sum(kVec)*sym.log(sum(kVec)/sum(prodlist))
        logTerm = [] #Verified
        for i in range(n):
            logTerm.append(kVec[i]*sym.log(prodlist[i]))
        thirdTerm = sum(logTerm)
        factTerm = [] #Verified
        for i in range(n):
            factTerm.append(sym.log(factorial(kVec[i])))
        fourthTerm = sum(factTerm)

        f = firstTerm + secondTerm + thirdTerm - fourthTerm
        return f, x

    def convertSym(self, x, bh):
        lambdify(x, bh, "numpy")

    def LLF(self, h, betas, covariate_data, n, kVec):
        # can clean this up to use less loops, probably
        covariate_num = len(betas)
        second = []
        prodlist = []
        for i in range(n):
            sum1=1
            sum2=1
            TempTerm1 = 1
            for j in range(covariate_num):
                    TempTerm1 = TempTerm1 * np.exp(covariate_data[j][i] * betas[j])
            #print('Test: ', TempTerm1)
            sum1=1-((1 - h[i]) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(covariate_num):
                        TempTerm2 = TempTerm2 * np.exp(covariate_data[j][k] * betas[j])
                #print ('Test:', TempTerm2)
                sum2 = sum2*((1 - h[i])**(TempTerm2))
            #print ('Sum2:', sum2)
            second.append(sum2)
            prodlist.append(sum1*sum2)

        firstTerm = -sum(kVec) #Verified
        secondTerm = sum(kVec)*np.log(sum(kVec)/sum(prodlist))
        logTerm = [] #Verified
        for i in range(n):
            logTerm.append(kVec[i]*np.log(prodlist[i]))
        thirdTerm = sum(logTerm)
        factTerm = [] #Verified
        for i in range(n):
            factTerm.append(np.log(np.math.factorial(kVec[i])))
        fourthTerm = sum(factTerm)

        return firstTerm + secondTerm + thirdTerm - fourthTerm

    def optimizeSolution(self, fd, B):
        return scipy.optimize.fsolve(fd, x0=B)

    def calcOmega(self, h, betas, covariateData, n, totalFailures):
        # can clean this up to use less loops, probably
        covariate_num = len(betas)
        prodlist = []
        for i in range(n):
            sum1=1
            sum2=1
            TempTerm1 = 1
            for j in range(covariate_num):
                    TempTerm1 = TempTerm1 * np.exp(covariateData[j][i] * betas[j])
            sum1=1-((1 - h[i]) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(covariate_num):
                        TempTerm2 = TempTerm2 * np.exp(covariateData[j][k] * betas[j])
                sum2 = sum2*((1 - h[i])**(TempTerm2))
            prodlist.append(sum1*sum2)
        denominator = sum(prodlist)
        numerator = totalFailures
        # print("numerator =", numerator, "denominator =", denominator)

        return numerator / denominator

    def calcP(self):
        pass

    def AIC(self, h, betas, covariate_data, n, kVec):
        p = 5   # why?
        return 2 * p - np.multiply(2, self.LLF(h, betas, covariate_data, n, kVec))

    def BIC(self, h, betas, covariate_data, n, kVec):
        p = 5   # why?
        return p * np.log(n) - 2 * self.LLF(h, betas, covariate_data, n, kVec)

    def MVF(self, h, omega, betas, covariate_data, n):
        # can clean this up to use less loops, probably
        covariate_num = len(betas)
        prodlist = []
        for i in range(n + 1):
            sum1=1
            sum2=1
            TempTerm1 = 1
            for j in range(covariate_num):
                    TempTerm1 = TempTerm1 * np.exp(covariate_data[j][i] * betas[j])
            sum1=1-((1 - h[i]) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(covariate_num):
                        TempTerm2 = TempTerm2 * np.exp(covariate_data[j][k] * betas[j])
                sum2 = sum2*((1 - h[i])**(TempTerm2))
            prodlist.append(sum1*sum2)
        return omega * sum(prodlist)

    def MVF_all(self, h, omega, betas, covariate_data, n):
        mvf_list = np.array([self.MVF(h, omega, betas, covariate_data, k) for k in range(n)])
        return mvf_list
    
    def SSE(self, fitted, actual):
        sub = np.subtract(fitted, actual)
        sse_error = np.sum(np.power(sub, 2))
        return sse_error

    def intensity_fit(self, mvfList):
        # first = np.array([mvf_list[0]])
        # difference = np.array(np.diff(mvf_list))
        # return np.concatenate(first, difference)  # want the same size as list that was input
        # print(mvf_list[0])
        difference = [mvfList[i+1]-mvfList[i] for i in range(len(mvfList)-1)]
        # print(difference, type(difference))
        return [mvfList[0]] + difference

    def model_fitting(srm="geometric"):
        # select which hazard function to use
        if (srm == "nb2"):
            # negative binomial (order 2)
            h = nb2_hazard
        elif (srm == "dw2"):
            # discrete weibull (order 2)
            h = dw2_hazard
        elif (srm == "nb"):
            # negative binomial
            h = nb_hazard
        elif (srm == "dw"):
            # discrete weibull
            h = dw_hazard
        else:
            # geometric is default
            h = geometric_hazard

        omega = calcOmega(h, betas, cov_sdata, n, total_failures)
        print("calculated omega =", omega)
        # omega = 41.627
        llf_val = LLF(h, betas, cov_data, n, kVec)      # log likelihood value
        aic_val = AIC(h, betas, cov_data, n, kVec)
        bic_val = BIC(h, betas, cov_data, n, kVec)
        mvf_list = MVF_all(h, omega, betas, cov_data, n)
        
        print("MVF values:", mvf_list)

        sse_val = SSE(mvf_list, kVec_cumulative)

        intensity_list = intensity_fit(mvf_list)
        print("intensity values:", intensity_list)