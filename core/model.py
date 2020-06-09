from abc import ABC, abstractmethod, abstractproperty

import logging as log

import time   # for testing

import numpy as np
import sympy as sym
from sympy import symbols, diff, exp, lambdify, DeferredVector, factorial, Symbol, Idx, IndexedBase
import scipy.optimize
from scipy.special import factorial as npfactorial

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
        self.numCovariates = len(self.covariateData)
        self.converged = False
        self.setupMetricString()

        # logging
        log.info("Failure times: %s", self.t)
        log.info("Number of time segments: %d", self.n)
        log.info("Failures: %s", self.failures)
        log.info("Cumulative failures: %s", self.cumulativeFailures)
        log.info("Total failures: %d", self.totalFailures)
        log.info("Number of covariates: %d", self.numCovariates)

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
    def shortName(self):
        """
        Shortened name of model (string)
        """
        return "Gen"

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

    def setupMetricString(self):
        if (self.metricNames == []):
            self.metricString = "None"
        else:
            self.metricString = ", ".join(self.metricNames)

    def symAll(self):
        """
        Called in mainWindow
        Creates symbolic LLF for model with all metrics, and differentiates
        """

        Model.maxCovariates = self.numCovariates    # UNNECESSARY, use self.data.numCovariates
        f, x = self.LLF_sym(self.hazardFunction)    # pass hazard rate function
        bh = np.array([diff(f, x[i]) for i in range(self.numCovariates + 1)])
        # Model.lambdaFunctionAll = self.convertSym(x, bh, "numpy")

        t1_start = time.process_time()
        f = self.convertSym(x, bh, "numpy")
        t1_stop = time.process_time()
        log.info("time to convert symbolic function: %s", t1_stop - t1_start)

        return f

    def LLF_sym(self, hazard):
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

    def LLF_sym_new(self, hazard):
        # fast, but incorrect
        x = DeferredVector('x')

        failures = np.array(self.failures)
        covariateData = np.array(self.covariateData)
        h = np.array([hazard(i, x[0]) for i in range(self.n)])


        failure_sum = np.sum(failures)

        term1 = np.sum(np.log(npfactorial(failures[i])) for i in range(self.n))
        term2 = failure_sum

        oneMinusB = np.array([sym.Pow((1.0 - hazard(i, x[0])), sym.prod([sym.exp(x[j] * covariateData[j - 1][i]) for j in range(1, self.numCovariates + 1)])) for i in range(self.n)])

        term3_num = np.sum(failures)
        term3_den1 = np.sum(np.subtract(1.0, oneMinusB))

        exponent = np.array([np.prod([[x[j] * covariateData[j - 1][k] for j in range(1, self.numCovariates + 1)] for k in range(i)]) for i in range(self.n)])

        product_array = np.array(np.power(np.subtract(1.0, h), exponent))

        term3_den2 = np.prod(product_array)
        term3 = sym.log(term3_num/(term3_den1 * term3_den2)) * failure_sum

        a = np.subtract(1.0, oneMinusB)
        b = a * term3_den2
        c = np.array([sym.log(b[i]) for i in range(b.shape[0])])
        d = np.multiply(c, failures)
        term4 = np.sum(d)

        f = -term1 - term2 + term3 + term4

        return f, x

    def convertSym(self, x, bh, target):
        return lambdify(x, bh, target)

    def runEstimation(self):
        initial = self.initialEstimates()

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
        log.info("Initial estimates: %s", initial)

        # fd = lambda a: m.lambdaFunctionAll(np.concatenate((a, zero_array), axis=0))
        fd = m.lambdaFunctionAll

        log.info("INITIAL ESTIMATES = %s", initial)
        # log.info(f"PASSING INITIAL ESTIMATES = {fd(initial)}")

        optimize_start = time.process_time()
        sol = self.optimizeSolution(fd, initial)
        optimize_stop = time.process_time()
        log.info("optimization time: %s", optimize_stop - optimize_start)
        log.info("Optimized solution: %s", sol)

        self.b = sol[0]
        self.betas = sol[1:]
        hazard = self.calcHazard(self.b, self.n)
        self.hazard = hazard    # for MVF prediction, don't want to calculate again
        self.modelFitting(hazard, self.betas)

    def initialEstimates(self):
        #return np.insert(np.random.uniform(min, max, self.numCovariates), 0, np.random.uniform(0.0, 0.1, 1)) #Works for GM and NB2
        # return np.insert(np.random.uniform(0.0, 0.01, self.numCovariates), 0, np.random.uniform(0.998, 0.99999,1))
                                                                    # (low, high, size)
                                                                    # size is numCovariates + 1 to have initial estimate for b
        betasEstimate = np.random.uniform(self.coxParameterEstimateRange[0], self.coxParameterEstimateRange[1], self.numCovariates)
        # print(self.shapeParameterEstimateRange)
        bEstimate = np.random.uniform(self.shapeParameterEstimateRange[0], self.shapeParameterEstimateRange[1], 1)
        return np.insert(betasEstimate, 0, bEstimate)   # insert b in the 0th location of betaEstimate array

    def optimizeSolution(self, fd, B):
        log.info("Solving for MLEs...")

        # solution = scipy.optimize.fsolve(fd, x0=B)

        try:
            log.info("Using broyden1")
            solution = scipy.optimize.broyden1(fd, xin=B, iter=100)
        except scipy.optimize.nonlin.NoConvergence:
            log.info("Using fsolve")
            solution = scipy.optimize.fsolve(fd, x0=B)
        except:
            log.info("Could Not Converge")
            solution = [0 for i in range(self.numCovariates + 1)]


        #solution = scipy.optimize.broyden2(fd, xin=B)          #Does not work (Seems to work well until the 3 covariates then crashes)
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

    def modelFitting(self, hazard, betas):
        self.omega = self.calcOmega(hazard, betas)
        log.info("Calculated omega: %s", self.omega)
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

    def AIC(self, h, betas):
        # +2 variables for any other algorithm
        p = len(betas) + 1 #+ 1   # number of covariates + number of hazard rate parameters + 1 (omega)
        return 2 * p - np.multiply(2, self.LLF(h, betas))

    def BIC(self, h, betas):
        # +2 variables for any other algorithm
        p = len(betas) + 1 #+ 1   # number of covariates + number of hazard rate parameters + 1 (omega)
        return p * np.log(self.n) - 2 * self.LLF(h, betas)

    def calcP(self):
        pass

    def MVF_all(self, h, omega, betas):
        mvf_array = np.array([self.MVF(h, self.omega, betas, dataPoints) for dataPoints in range(self.n)])
        return mvf_array

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

    def SSE(self, fitted, actual):
        sub = np.subtract(fitted, actual)
        sseError = np.sum(np.power(sub, 2))
        return sseError

    def intensityFit(self, mvf_array):
        difference = [mvf_array[i+1]-mvf_array[i] for i in range(len(mvf_array)-1)]
        return [mvf_array[0]] + difference

    def prediction(self, failures):
        total_points = self.n + failures
        zero_array = np.zeros(failures) # to append to existing covariate data
        new_covData = [0 for i in range(self.numCovariates)]

        hazard = self.calcHazard(self.b, total_points)  # calculate new values for hazard function

        for j in range(self.numCovariates):
            new_covData[j] = np.append(self.covariateData[j], zero_array)

        mvf_array = np.array([self.MVF_prediction(new_covData, hazard, dataPoints) for dataPoints in range(total_points)])
        intensity_array = self.intensityFit(mvf_array)
        x = np.arange(0, total_points + 1)

        # add initial point at zero if not present
        if self.t[0] != 0:
            mvf_array = np.concatenate((np.zeros(1), mvf_array))
            intensity_array = np.concatenate((np.zeros(1), intensity_array))

        return (x, mvf_array, intensity_array)

    def MVF_prediction(self, covariateData, hazard, stop):
        # can clean this up to use less loops, probably
        prodlist = []
        for i in range(stop + 1):     # CHANGED THIS FROM self.n + 1 !!!
            sum1 = 1
            sum2 = 1
            TempTerm1 = 1
            for j in range(self.numCovariates):
                TempTerm1 = TempTerm1 * np.exp(covariateData[j][i] * self.betas[j])
            sum1 = 1-((1 - hazard[i]) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(self.numCovariates):
                    TempTerm2 = TempTerm2 * np.exp(covariateData[j][k] * self.betas[j])
                sum2 = sum2 * ((1 - hazard[i])**(TempTerm2))
            prodlist.append(sum1 * sum2)
        return self.omega * sum(prodlist)

    def allocationFunction(self, x, *args):
        failures = args[0]
        # i = self.n + failures
        i = self.n
        return -(self.MVF_allocation(self.hazardFunction, self.omega, self.betas, i, x))    # must be negative, SHGO uses minimization

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
