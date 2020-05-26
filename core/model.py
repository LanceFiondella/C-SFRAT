from abc import ABC, abstractmethod, abstractproperty

import logging as log

import time   # for testing

import numpy as np
import sympy as sym
from sympy import symbols, diff, exp, lambdify, DeferredVector, factorial, Symbol, Idx, IndexedBase
import scipy.optimize
from scipy.special import factorial

import models   # maybe??

class Model(ABC):

    # lambdaFunctionAll = None
    maxCovariates = None

    def __init__(self, *args, **kwargs):
        """
        Initialize Model

        Keyword Args:
            data: Pandas dataframe with all required columns
            metrics: list of selected metric names
        """
        self.data = kwargs["data"]                  # dataframe
        self.metricNames = kwargs["metricNames"]    # selected metric names (strings)
        self.t = self.data.iloc[:, 0].values            # failure times, from first column of dataframe
        self.failures = self.data.iloc[:, 1].values     # number of failures, from second column of dataframe
        self.n = len(self.failures)                     # number of discrete time segments
        self.cumulativeFailures = self.data["CFC"].values
        self.totalFailures = self.cumulativeFailures[-1]
        # list of arrays or array of arrays?
        self.covariateData = [self.data[name].values for name in self.metricNames]
        # log.info(f"covariate data = {self.covariateData}")
        # log.info(f"")
        self.numCovariates = len(self.covariateData) 
        self.converged = False
        self.setupMetricString()

        # logging
        # log.info("Failure times: %s", self.t)
        # log.info("Number of time segments: %d", self.n)
        # log.info("Failures: %s", self.failures)
        # log.info("Cumulative failures: %s", self.cumulativeFailures)
        # log.info("Total failures: %d", self.totalFailures)
        # log.info("Number of covariates: %d", self.numCovariates)

    ################################################
    # Properties/Members all models must implement #
    ################################################
    @property
    @abstractmethod
    def name(self):
        """
        Name of model (string)
        """
        return "Generic Model"

    @property
    @abstractmethod
    def coxParameterEstimateRange(self):
        """
        Define Cox parameter estimate range for root finding initial values
        """
        return [0.0, 0.01]

    @property
    @abstractmethod
    def shapeParameterEstimateRange(self):
        """
        Define shape parameter estimate range for root finding initial values
        """
        return [0.0, 0.1]

    # @property
    # @abstractmethod
    # def symbolicDifferentiation(self):
    #     """
    #     Set False if manually implementing log-likelihood function and its derivative
    #     """
    #     return True

    ##################################################
    # Methods that must be implemented by all models #
    ##################################################

    @abstractmethod
    def calcHazard(self):
        pass

    @abstractmethod
    def hazardFunction(self):
        pass

    # @abstractmethod
    # def runEstimation(self):
    #     """
    #     main method that calls others; called by TaskThread
    #     """
    #     pass

    def setupMetricString(self):
        if (self.metricNames == []):
            self.metricString = "None"
        else:
            self.metricString = ", ".join(self.metricNames)

    def initialEstimates(self):
        #return np.insert(np.random.uniform(min, max, self.numCovariates), 0, np.random.uniform(0.0, 0.1, 1)) #Works for GM and NB2
        # return np.insert(np.random.uniform(0.0, 0.01, self.numCovariates), 0, np.random.uniform(0.998, 0.99999,1))
                                                                    # (low, high, size)
                                                                    # size is numCovariates + 1 to have initial estimate for b
        betasEstimate = np.random.uniform(self.coxParameterEstimateRange[0], self.coxParameterEstimateRange[1], self.numCovariates)
        # print(self.shapeParameterEstimateRange)
        bEstimate = np.random.uniform(self.shapeParameterEstimateRange[0], self.shapeParameterEstimateRange[1], 1)
        return np.insert(betasEstimate, 0, bEstimate)   # insert b in the 0th location of betaEstimate array

    def LLF_sym_old(self, hazard):
        # x[0] = b
        # x[1:] = beta1, beta2, ..

        x = DeferredVector('x')
        second = []
        prodlist = []
        for i in range(self.n):
            sum1 = 1
            sum2 = 1
            TempTerm1 = 1
            for j in range(1, self.numCovariates + 1):
                TempTerm1 = TempTerm1 * exp(self.covariateData[j - 1][i] * x[j])
            sum1 = 1 - ((1 - (hazard(i, x[0]))) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(1, self.numCovariates + 1):
                    TempTerm2 = TempTerm2 * exp(self.covariateData[j - 1][k] * x[j])
                sum2 = sum2 * ((1 - (hazard(i, x[0])))**(TempTerm2))
            second.append(sum2)
            prodlist.append(sum1*sum2)

        firstTerm = -sum(self.failures) #Verified
        secondTerm = sum(self.failures)*sym.log(sum(self.failures)/sum(prodlist))
        logTerm = [] #Verified
        for i in range(self.n):
            logTerm.append(self.failures[i]*sym.log(prodlist[i]))
        thirdTerm = sum(logTerm)
        factTerm = [] #Verified
        for i in range(self.n):
            factTerm.append(sym.log(factorial(self.failures[i])))
        fourthTerm = sum(factTerm)

        f = firstTerm + secondTerm + thirdTerm - fourthTerm
        return f, x

    def LLF_sym(self, hazard):
        x = DeferredVector('x')

        failures = np.array(self.failures)
        covariateData = np.array(self.covariateData)
        h = np.array([hazard(i, x[0]) for i in range(self.n)])


        failure_sum = np.sum(failures)

        term1 = np.sum(np.log(np.math.factorial(failures[i])) for i in range(self.n))
        term2 = failure_sum

        oneMinusB = np.array([sym.Pow((1.0 - hazard(i, x[0])), sym.prod([sym.exp(x[j] * covariateData[j - 1][i]) for j in range(1, self.numCovariates + 1)])) for i in range(self.n)])
        # print(oneMinusB)

        # oneMinusB = oneMinusB.reshape((oneMinusB.shape[0],))

        term3_num = np.sum(failures)
        term3_den1 = np.sum(np.subtract(1.0, oneMinusB))


        # for i in range(self.n):
        #     for k in range(i):
        #         for j in range(self.numCovariates):
        #             np.power((1.0 - h[i]), np.array(np.exp(betas[j] * self.covariateData[j][k])))

        exponent = np.array([np.prod([[x[j] * covariateData[j - 1][k] for j in range(1, self.numCovariates + 1)] for k in range(i)]) for i in range(self.n)])
        # print(exponent)

        # np.array([np.product([[betas[j] * cov[j][k] for j in range(3)] for k in range(i)]) for i in range(15)])

        product_array = np.array(np.power(np.subtract(1.0, h), exponent))
        # print(product_array)
        # product_array = np.array([(np.power((1.0 - h[i]), np.array([np.exp(betas[j] * self.covariateData[j][k]) for j in range(self.numCovariates)])) for k in range(i)) for i in range(self.n)])
        # print(type(product_array))
        term3_den2 = np.prod(product_array)
        term3 = sym.log(term3_num/(term3_den1 * term3_den2)) * failure_sum


        # print(oneMinusB)
        # print(term3_den2)
        # print(self.failures)

        a = np.subtract(1.0, oneMinusB)
        # print("a =", a, "of type", type(a))
        # print("term3_den2 =", term3_den2, "of type", type(term3_den2))
        # b = np.prod(a, term3_den2)
        b = a * term3_den2
        # print(b[0])
        c = np.array([sym.log(b[i]) for i in range(b.shape[0])])
        # print("c =", c, "of type", type(c))
        # print("failures =", failures, "of type", type(failures))
        d = np.multiply(c, failures)
        term4 = np.sum(d)

        # term4 = np.sum(np.prod(np.log(np.prod(np.subtract(1.0, oneMinusB), term3_den2)), np.array(self.failures)))

        f = -term1 - term2 + term3 + term4


        return f, x

    def LLF_sym_new2(self, hazard):
        x = DeferredVector('x')

        failures = np.array(self.failures)
        covariateData = np.array(self.covariateData)

        h = np.array([hazard(i, x[0]) for i in range(self.n)])


        failure_sum = np.sum(failures)

        term1 = np.sum(np.log(factorial(failures)))
        term2 = failure_sum

        oneMinusB_array = np.subtract(1.0, h)

        covTransposed = covariateData.transpose()
        # exponent_product = np.zeros((len(covTransposed), self.numCovariates))
        exponent_product = [[0 for j in range(self.numCovariates)] for i in range(len(covTransposed))]

        # print(x[1] * covTransposed[0][0])

        for i in range(self.n):
            for j in range(self.numCovariates):
                exponent_product[i][j] = x[j + 1] * covTransposed[i][j]
            exponent_product[i] = np.sum(exponent_product[i])

        # print(exponent_product)

        # exponent = np.exp(np.array(exponent_product))
        exponent = np.array([sym.exp(exponent_product[i]) for i in range(len(exponent_product))])
        # print(exponent)

        denom1 = np.power(oneMinusB_array, exponent)

        term3_num = np.sum(failures)
        term3_den1_array = np.subtract(1.0, denom1)

        ########

        # for k in range(self.n):
        #     for i in range(k):
        #         for j in range(self.numCovariates):
        #             exponent_product[i][j] = x[j + 1] * covTransposed[i][j]
        #         exponent_product[i] = np.sum(exponent_product[i])

        # term3_den2 = np.zeros(self.n)
        term3_den2 = [0 for i in range(self.n)]
        for k in range(self.n):
            # already calculated (1 - b)^(exp(beta*cov))
            # can just use slicing for that array
            # print(denom1[0:k + 1])
            # subset = denom1[0:k + 1]
            # prod_temp = 1
            # prod_temp = [(prod_temp * i) for i in subset]
            term3_den2[k] = np.prod(denom1[0:k + 1])
            # print(prod_temp)
            # term3_den2[k] = prod_temp

        # exponent = np.array([np.prod([[x[j] * covariateData[j - 1][k] for j in range(1, self.numCovariates + 1)] for k in range(i)]) for i in range(self.n)])

        product_array = np.multiply(term3_den1_array, term3_den2)
        term3_den = np.sum(product_array)

        term3 = sym.log(term3_num/term3_den) * failure_sum

        # a = np.subtract(1.0, oneMinusB)
        # b = a * term3_den2
        # c = np.array([sym.log(b[i]) for i in range(b.shape[0])])
        # d = np.multiply(c, failures)
        # term4 = np.sum(d)

        # print(type(product_array))
        log_term = [sym.log(i) for i in product_array]
        a = np.array(log_term) * failures
        # a = np.log(product_array) * failures
        term4 = np.sum(a)
        
        f = -term1 - term2 + term3 + term4

        return f, x

    def LLF_sym_new3(self, hazard):
        x = DeferredVector('x')

        failures = np.array(self.failures)
        covariateData = np.array(self.covariateData)
        h = np.array([hazard(i, x[0]) for i in range(self.n)])


        failure_sum = np.sum(failures)

        term1 = np.sum(np.log(np.math.factorial(failures[i])) for i in range(self.n))
        term2 = failure_sum

        oneMinusB = np.array([sym.Pow((1.0 - hazard(i, x[0])), sym.prod([sym.exp(x[j] * covariateData[j - 1][i]) for j in range(1, self.numCovariates + 1)])) for i in range(self.n)])

        term3_num = np.sum(failures)
        # term3_den1 = np.sum(np.subtract(1.0, oneMinusB))

        # exponent = np.array([np.prod([[x[j] * covariateData[j - 1][k] for j in range(1, self.numCovariates + 1)] for k in range(i)]) for i in range(self.n)])
        term3_den2_array = np.array([np.product(oneMinusB[0:k + 1]) for k in range(self.n)])
        term3_den1_array = np.subtract(1, oneMinusB)

        product_array = term3_den1_array * term3_den2_array

        term3 = sym.log(term3_num/np.sum(product_array)) * failure_sum

        # a = np.subtract(1.0, oneMinusB)
        # b = a * term3_den2
        # c = np.array([sym.log(b[i]) for i in range(b.shape[0])])
        # d = np.multiply(c, failures)
        # term4 = np.sum(d)


        a = np.array([sym.log(i) for i in product_array])
        b = a * failures

        term4 = np.sum(b)

        f = -term1 - term2 + term3 + term4

        return f, x

    def newLLF(self, h, betas):
        # term1 = np.sum(np.log(np.math.factorial(self.failures[i])) for i in range(self.n))
        # term2 = np.sum(self.failures)
        # term3_num = np.sum(self.failures)
        # term3_den1 = np.sum(1 - np.power((1.0 - betas[0]), (np.exp(betas[j] * self.covariateData[j][i]) for j in range(self.numCovariates))) for i in range(self.n))
        # term3_den2 = np.prod((np.power((1.0 - betas[0]), (np.exp(betas[j] * self.covariateData[j][k]) for j in range(self.numCovariates))) for k in range(i)) for i in range(self.n))
        # term4_1 = np.sum(np.log((1 - np.power((1 - betas[0]), (np.exp(betas[j] * self.covariateData[j][i]) for j in range(self.numCovariates))) for i in range(self.n)) * term3_den2))
        # # term4_2

        # return -term1 - term2 + np.log(term3_num/(term3_den1 * term3_den2)) * term2 + 
        failures = np.array(self.failures)
        betas = np.array(self.betas)
        covariateData = np.array(self.covariateData)


        failure_sum = np.sum(failures)

        term1 = np.sum(np.log(np.math.factorial(failures[i])) for i in range(self.n))
        term2 = failure_sum

        oneMinusB = np.array([np.power((1.0 - h[i]), np.prod(np.array([np.exp(betas[j] * covariateData[j][i]) for j in range(self.numCovariates)]))) for i in range(self.n)])
        # print(oneMinusB)

        # oneMinusB = oneMinusB.reshape((oneMinusB.shape[0],))

        term3_num = np.sum(failures)
        term3_den1 = np.sum(np.subtract(1.0, oneMinusB))


        # for i in range(self.n):
        #     for k in range(i):
        #         for j in range(self.numCovariates):
        #             np.power((1.0 - h[i]), np.array(np.exp(betas[j] * self.covariateData[j][k])))

        exponent = np.array([np.prod([[betas[j] * covariateData[j][k] for j in range(self.numCovariates)] for k in range(i)]) for i in range(self.n)])
        # print(exponent)

        # np.array([np.product([[betas[j] * cov[j][k] for j in range(3)] for k in range(i)]) for i in range(15)])

        product_array = np.array(np.power(np.subtract(1.0, h), exponent))
        # print(product_array)
        # product_array = np.array([(np.power((1.0 - h[i]), np.array([np.exp(betas[j] * self.covariateData[j][k]) for j in range(self.numCovariates)])) for k in range(i)) for i in range(self.n)])
        # print(type(product_array))
        term3_den2 = np.prod(product_array)
        term3 = np.log(term3_num/(term3_den1 * term3_den2)) * failure_sum


        # print(oneMinusB)
        # print(term3_den2)
        # print(self.failures)

        a = np.subtract(1.0, oneMinusB)
        # print("a =", a, "of type", type(a))
        # print("term3_den2 =", term3_den2, "of type", type(term3_den2))
        # b = np.prod(a, term3_den2)
        b = a * term3_den2
        c = np.log(b)
        # print("c =", c, "of type", type(c))
        # print("failures =", failures, "of type", type(failures))
        d = np.multiply(c, failures)
        term4 = np.sum(d)

        # term4 = np.sum(np.prod(np.log(np.prod(np.subtract(1.0, oneMinusB), term3_den2)), np.array(self.failures)))

        return -term1 - term2 + term3 + term4

    def convertSym(self, x, bh, target):
        return lambdify(x, bh, target)

    def LLF(self, h, betas):
        # can clean this up to use less loops, probably
        second = []
        prodlist = []
        for i in range(self.n):
            sum1 = 1
            sum2 = 1
            TempTerm1 = 1
            for j in range(self.numCovariates):
                TempTerm1 = TempTerm1 * np.exp(self.covariateData[j][i] * betas[j])
            sum1 = 1 - ((1 - h[i]) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(self.numCovariates):
                    TempTerm2 = TempTerm2 * np.exp(self.covariateData[j][k] * betas[j])
                sum2 = sum2*((1 - h[i])**(TempTerm2))
            second.append(sum2)
            prodlist.append(sum1*sum2)

        firstTerm = -sum(self.failures) #Verified
        secondTerm = sum(self.failures)*np.log(sum(self.failures)/sum(prodlist))
        logTerm = [] #Verified
        for i in range(self.n):
            logTerm.append(self.failures[i]*np.log(prodlist[i]))
        thirdTerm = sum(logTerm)
        factTerm = [] #Verified
        for i in range(self.n):
            factTerm.append(np.log(np.math.factorial(self.failures[i])))
        fourthTerm = sum(factTerm)

        return firstTerm + secondTerm + thirdTerm - fourthTerm

    def optimizeSolution(self, fd, B):
        log.info("Solving for MLEs...")

        solution = scipy.optimize.fsolve(fd, x0=B)

        # try:
        #     log.info("Using broyden1")
        #     solution = scipy.optimize.broyden1(fd, xin=B, iter=250)
        # except scipy.optimize.nonlin.NoConvergence:
        #     log.info("Using fsolve")
        #     solution = scipy.optimize.fsolve(fd, x0=B)
        # except:
        #     log.info("Could Not Converge")
        #     solution = [0 for i in range(self.numCovariates + 1)]


        # solution = scipy.optimize.broyden2(fd, xin=B)          #Does not work (Seems to work well until the 3 covariates then crashes)
        #solution = scipy.optimize.anderson(fd, xin=B)          #Works for DW2 - DS1  - EstB{0.998, 0.999} Does not work for DS2
        #solution = scipy.optimize.excitingmixing(fd, xin=B)    #Does not work

        #solution = scipy.optimize.newton_krylov(fd, xin=B)     #Does not work

        #solution = scipy.optimize.linearmixing(fd, xin=B)      #Does not work
        #solution = scipy.optimize.diagbroyden(fd, xin=B)       #Does not Work

        #solution = scipy.optimize.root(fd, x0=B, method='hybr')
        #solution = scipy.optimize.fsolve(fd, x0=B)
        log.info("MLEs solved.")
        log.info("Solution: %s", solution)
        return solution

    def calcOmega(self, h, betas):
        # can clean this up to use less loops, probably
        prodlist = []
        for i in range(self.n):
            sum1 = 1
            sum2 = 1
            TempTerm1 = 1
            for j in range(self.numCovariates):
                    TempTerm1 = TempTerm1 * np.exp(self.covariateData[j][i] * betas[j])
            sum1 = 1-((1 - h[i]) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(self.numCovariates):
                        TempTerm2 = TempTerm2 * np.exp(self.covariateData[j][k] * betas[j])
                sum2 = sum2*((1 - h[i])**(TempTerm2))
            prodlist.append(sum1*sum2)
        denominator = sum(prodlist)
        numerator = self.totalFailures

        return numerator / denominator

    def calcP(self):
        pass

    def AIC(self, h, betas):
        # +2 variables for any other algorithm
        p = len(betas) + 1 #+ 1   # number of covariates + number of hazard rate parameters + 1 (omega)
        return 2 * p - np.multiply(2, self.LLF(h, betas))

    def BIC(self, h, betas):
        # +2 variables for any other algorithm
        p = len(betas) + 1 #+ 1   # number of covariates + number of hazard rate parameters + 1 (omega)
        return p * np.log(self.n) - 2 * self.LLF(h, betas)

    def SSE(self, fitted, actual):
        sub = np.subtract(fitted, actual)
        sseError = np.sum(np.power(sub, 2))
        return sseError

    def MVF(self, h, omega, betas, stop):
        # can clean this up to use less loops, probably
        prodlist = []
        for i in range(stop + 1):     # CHANGED THIS FROM self.n + 1 !!!
            sum1 = 1
            sum2 = 1
            TempTerm1 = 1
            for j in range(self.numCovariates):
                TempTerm1 = TempTerm1 * np.exp(self.covariateData[j][i] * betas[j])
            sum1 = 1-((1 - h[i]) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(self.numCovariates):
                    TempTerm2 = TempTerm2 * np.exp(self.covariateData[j][k] * betas[j])
                sum2 = sum2 * ((1 - h[i])**(TempTerm2))
            prodlist.append(sum1 * sum2)
        return omega * sum(prodlist)

    def MVF_new(self, h, omega, betas, stop):
        betas = np.array(self.betas[0:self.numCovariates])
        print("STOP =", stop)

        term2 = []  # need to append, use list first then convert to np array after

        if self.numCovariates == 0:
            oneMinusB_array = np.subtract(1.0, h[0:stop + 1])
            term1 = np.subtract(1.0, oneMinusB_array)

            for k in range(stop + 1):
                oneMinusB_array = np.subtract(1.0, h[0:k + 1])
                term2.append(np.prod(oneMinusB_array))
        else:
            covArray = np.array(self.covariateData)
            print("COV DATA =", covArray)
            covariateData = np.array(covArray[:, 0:stop + 1])
            print("NEW COVARIATE DATA =", covariateData)
            covTransposed = covariateData.transpose()

            temp = np.zeros(len(covTransposed))
            for i in range(len(covTransposed)):
                temp[i] = np.sum(betas * covTransposed[i])

            # np.multiply(betas, covariateData)

            exponent1 = np.exp(temp)
            # oneMinusB_array = np.array([1.0 - h[i]] for i in range(stop + 1))
            oneMinusB_array = np.subtract(1.0, h[:stop + 1])
            term1 = np.subtract(1.0, np.power(oneMinusB_array, exponent1))

            for k in range(stop + 1):
                cov_subset = covariateData[:, 0:k+1]
                subset_transposed = cov_subset.transpose()
                print("k cov data =", cov_subset)
                print("transposed =", subset_transposed)
                # print("transpose =", np.transpose(np.array([betas,] * covariateData[:, 0:k+1].shape[0])))
                # res = covariateData[:, 0:k] * np.transpose(np.array([betas,] * covariateData[:, 0:k+1].shape[0]))

                # res = np.multiply(betas[:, None], covariateData[:, 0:k+1])
                res = np.zeros(len(subset_transposed))
                print(len(subset_transposed))
                for i in range(len(subset_transposed)):
                    res[i] = np.sum(betas * subset_transposed[i])

                # res = np.zeros(self.numCovariates)

                # for i in range(len(betas)):
                #     # res = 0
                #     # res = res + betas[i] * 
                #     res[i] = betas[i] * covariateData[i, 0:k+1]

                # for i in range(len(covariateData[:, 0:k+1])):
                #     res = betas[1]

                # exponent2 = np.exp(np.multiply(betas, covariateData[:, 0:k]))
                exponent2 = np.exp(res)
                oneMinusB_array = np.subtract(1.0, h[0:k+1])
                print(oneMinusB_array)
                print(exponent2)
                powerTerm = np.power(oneMinusB_array, exponent2)
                term2.append(np.prod(powerTerm))

        # term2 = np.array(term2)

        t1 = term1 * np.prod(term2)

        return omega * np.sum(t1)

        # t1 = np.sum(term1)
        # t2 = np.sum(term2)

        # exponent1 = np.prod(np.array([np.exp(betas[j] * covariateData[j][i]) for j in range(self.numCovariates)]))
        # oneMinusB = np.array([np.power((1.0 - h[i]), exponent1) for i in range(stop + 1)])
        # term3_den1 = np.sum(np.subtract(1.0, oneMinusB))
        # exponent = np.array([np.prod([[betas[j] * covariateData[j][k] for j in range(self.numCovariates)] for k in range(i-1)]) for i in range(stop + 1)])

        # product_array = np.array(np.power(np.subtract(1.0, h[:stop + 1]), exponent))
        # term3_den2 = np.prod(product_array)
        # term3 = term3_den1 * term3_den2

        # return omega * np.sum(term3)

        # return omega * t1 * t2

    def MVF_all(self, h, omega, betas):
        mvfList = np.array([self.MVF(h, self.omega, betas, dataPoints) for dataPoints in range(self.n)])
        return mvfList

    def MVF_allocation(self, h, omega, betas, stop, x):
        """
        x is vector of covariate metrics chosen for allocation
        """
        # can clean this up to use less loops, probably
        covData = [list(self.covariateData[j]) for j in range(self.numCovariates)]
        
        for j in range(self.numCovariates):
            covData[j].append(x[j]) # append a single variable (x[j]) to the end of each vector of covariate data

        prodlist = []
        for i in range(stop + 1):     # CHANGED THIS FROM self.n + 1 !!!
            sum1 = 1
            sum2 = 1
            TempTerm1 = 1
            for j in range(self.numCovariates):
                TempTerm1 = TempTerm1 * np.exp(covData[j][i] * betas[j])
            sum1 = 1-((1 - h(i, self.b)) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(self.numCovariates):
                    TempTerm2 = TempTerm2 * np.exp(covData[j][k] * betas[j])
                sum2 = sum2 * ((1 - h(i, self.b))**(TempTerm2))
            prodlist.append(sum1 * sum2)
        return (omega * sum(prodlist))
    
    def allocationFunction(self, x, *args):
        failures = args[0]
        # i = self.n + failures
        i = self.n
        return -(self.MVF_allocation(self.hazardFunction, self.omega, self.betas, i, x))    # must be negative, SHGO uses minimization

    def intensityFit(self, mvfList):
        difference = [mvfList[i+1]-mvfList[i] for i in range(len(mvfList)-1)]
        return [mvfList[0]] + difference

    def runEstimation(self):
        initial = self.initialEstimates()
        # log.info("Initial estimates: %s", initial)
        # f, x = self.LLF_sym(self.hazardFunction)    # pass hazard rate function
        # bh = np.array([diff(f, x[i]) for i in range(self.numCovariates + 1)])
        # log.info("Log-likelihood differentiated.")
        # log.info("Converting symbolic equation to numpy...")
        # fd = self.convertSym(x, bh, "numpy")


        # b, beta1, beta2, beta3
        # log.info("fd after convert = %s", fd)

        # need class of specific model being used, lambda function stored as class variable
        # log.info("name = %s", self.__class__.__name__)
        m = models.modelList[self.__class__.__name__]

        # ex. (max covariates = 3) for 3 covariates, zero_array should be length 0
        # for no covariates, zero_array should be length 3
        numZeros = Model.maxCovariates - self.numCovariates
        zero_array = np.zeros(numZeros)   # create empty array, size of num covariates
        # create new lambda function that calls lambda function for all covariates
        # for no covariates, concatenating array a with zero element array produces a



        initial = np.concatenate((initial, zero_array), axis=0)

        # fd = lambda a: m.lambdaFunctionAll(np.concatenate((a, zero_array), axis=0))
        fd = m.lambdaFunctionAll






        log.info("INITIAL ESTIMATES = %s", initial)
        # log.info(f"PASSING INITIAL ESTIMATES = {fd(initial)}")

        t1_start = time.process_time()

        sol = self.optimizeSolution(fd, initial)
        log.info("Optimized solution: %s", sol)

        t1_stop = time.process_time()
        print("optimization time:", t1_stop - t1_start)

        self.b = sol[0]
        self.betas = sol[1:]
        hazard = self.calcHazard(self.b)
        self.modelFitting(hazard, self.betas)

    def modelFitting(self, hazard, betas):
        self.omega = self.calcOmega(hazard, betas)
        log.info("Calculated omega: %s", self.omega)

        # t1_start = time.process_time()
        # for i in range(10000):
        #     self.llfVal = self.LLF(hazard, betas)      # log likelihood value
        # t1_stop = time.process_time()
        # log.info("Calculated log-likelihood value: %s", self.llfVal)
        # print("original elapsed time =", t1_stop - t1_start)

        # t2_start = time.process_time()
        # for i in range(10000):
        #     newLLFval = self.newLLF(hazard, betas)
        # t2_stop = time.process_time()
        # log.info("New log-likelihood value: %s", newLLFval)
        # print("new elapsed time =", t2_stop - t2_start)

        self.llfVal = self.LLF(hazard, betas)      # log likelihood value
        log.info("Calculated log-likelihood value: %s", self.llfVal)

        self.aicVal = self.AIC(hazard, betas)
        log.info("Calculated AIC: %s", self.aicVal)
        self.bicVal = self.BIC(hazard, betas)
        log.info("Calculated BIC: %s", self.bicVal)
        self.mvfList = self.MVF_all(hazard, self.omega, betas)

        # temporary
        if (np.isnan(self.llfVal) or np.isinf(self.llfVal)):
            self.converged = False
        else:
            self.converged = True

        self.sseVal = self.SSE(self.mvfList, self.cumulativeFailures)
        log.info("Calculated SSE: %s", self.sseVal)
        self.intensityList = self.intensityFit(self.mvfList)

        log.info("MVF values: %s", self.mvfList)
        log.info("Intensity values: %s", self.intensityList)

    def symAll(self):
        """
        Called in mainWindow
        Creates symbolic LLF for model with all metrics, and differentiates
        """
        # f, x = self.LLF_sym(self.hazardFunction)    # pass hazard rate function
        # # bh = np.array([diff(f, x[i]) for i in range(self.numCovariates + 1)])
        # # for i in range(self.numCovariates):
        # bh = [0 for i in range(self.numCovariates)]
        # # for i in range(self.numCovariates, 0, -1):
        # #     # n, n-1, ..., 3, 2, 1 [NOT 0]
        # #     # x[1:i+1] = 0
        # #     bh[i-1] = np.array([diff(f, x[i]) for i in range(self.numCovariates + 1)])
        # #     x[i:self.numCovariates + 1] = 0
        # for i in range(self.numCovariates):
        #     x_copy = x.subs()
        #     log.info("x = {0}".format(x))
        #     x_copy[1:i+1] = [0 for j in range(i)]
        #     log.info("x_copy = {0}".format(x_copy))
        #     bh[i] = np.array([diff(f, x_copy[j]) for j in range(self.numCovariates + 1)])
        #     log.info("bh[{0}] = {1}".format(i, bh[i]))

        t1_start = time.process_time()
        Model.maxCovariates = self.numCovariates    # UNNECESSARY, use self.data.numCovariates
        f, x = self.LLF_sym(self.hazardFunction)    # pass hazard rate function
        bh = np.array([diff(f, x[i]) for i in range(self.numCovariates + 1)])
        # Model.lambdaFunctionAll = self.convertSym(x, bh, "numpy")

        # return self.convertSym(x, bh, "numpy")

        f = self.convertSym(x, bh, "numpy")
        t1_stop = time.process_time()
        print("time to convert:", t1_stop - t1_start)
        return f


        # lambdaFunctions = [0 for i in range(self.numCovariates)]
        # lambdaFunctions[self.numCovariates - 1] = self.convertSym(x, bh, "numpy")
        # for i in range(self.numCovariates - 1, -1, -1):
        #     # lambdaFunctions[i] = 
        #     print(lambdaFunctions[2]([0, 0, 0, 0]))