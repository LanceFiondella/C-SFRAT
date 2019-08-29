from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import sympy as sym
from sympy import symbols, exp, lambdify, DeferredVector, factorial, Symbol, Idx, IndexedBase
import scipy.optimize

import logging

class Model(ABC):
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
        self.cumulativeFailures = self.data["Cumulative"].values
        self.totalFailures = self.cumulativeFailures[-1]
        # list of arrays or array of arrays?
        self.covariateData = [self.data[name].values for name in self.metricNames]
        self.numCovariates = len(self.covariateData)

        # logging
        logging.info("Failure times: {0}".format(self.t))
        logging.info("Number of time segments: {0}".format(self.n))
        logging.info("Failures: {0}".format(self.failures))
        logging.info("Cumulative failures: {0}".format(self.cumulativeFailures))
        logging.info("Total failures: {0}".format(self.totalFailures))
        logging.info("Number of covariates: {0}".format(self.numCovariates))

    ##############################################
    #Properties/Members all models must implement#
    ##############################################
    @property
    @abstractmethod
    def name(self):
        """
        Name of model (string)
        """
        return "Generic Model"

    @property
    @abstractmethod
    def converged(self):
        """
        Indicates whether model has converged (Bool)
        Must be set after parameters are calculated
        """
        return False

    ################################################
    #Methods that must be implemented by all models#
    ################################################
    @abstractmethod
    def calcHazard(self):
        pass

    # @abstractmethod
    # def modelFitting(self):
    #     pass

    @abstractmethod
    def runEstimation(self):
        """
        main method that calls others; called by TaskThread
        """
        pass

    def initialEstimates(self):
        return np.random.uniform(0.0, 0.1, self.numCovariates + 1)

    def LLF_sym(self):
        #Equation (30)
        x = DeferredVector('x')
        second = []
        prodlist = []
        for i in range(self.n):
            sum1 = 1
            sum2 = 1
            TempTerm1 = 1
            for j in range(1, self.numCovariates + 1):
                    TempTerm1 = TempTerm1 * exp(self.covariateData[j - 1][i] * x[j])
            #print('Test: ', TempTerm1)
            sum1 = 1 - ((1-x[0]) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(1, self.numCovariates + 1):
                        TempTerm2 = TempTerm2 * exp(self.covariateData[j - 1][k] * x[j])
                #print ('Test:', TempTerm2)
                sum2 = sum2 * ((1 - x[0])**(TempTerm2))
            #print ('Sum2:', sum2)
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
            #print('Test: ', TempTerm1)
            sum1 = 1 - ((1 - h[i]) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(self.numCovariates):
                        TempTerm2 = TempTerm2 * np.exp(self.covariateData[j][k] * betas[j])
                #print ('Test:', TempTerm2)
                sum2 = sum2*((1 - h[i])**(TempTerm2))
            #print ('Sum2:', sum2)
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
        return scipy.optimize.fsolve(fd, x0=B)

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
        # print("numerator =", numerator, "denominator =", denominator)

        return numerator / denominator

    def calcP(self):
        pass

    def AIC(self, h, betas):
        p = 5   # why?
        return 2 * p - np.multiply(2, self.LLF(h, betas))

    def BIC(self, h, betas):
        p = 5   # why?
        return p * np.log(self.n) - 2 * self.LLF(h, betas)

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

    def MVF_all(self, h, omega, betas):
        mvfList = np.array([self.MVF(h, omega, betas, k) for k in range(self.n)])
        return mvfList
    
    def SSE(self, fitted, actual):
        sub = np.subtract(fitted, actual)
        sseError = np.sum(np.power(sub, 2))
        return sseError

    def intensityFit(self, mvfList):
        difference = [mvfList[i+1]-mvfList[i] for i in range(len(mvfList)-1)]
        return [mvfList[0]] + difference

    def modelFitting(self, hazard, betas):
        omega = self.calcOmega(hazard, betas)
        logging.info("Calculated omega: {0}".format(omega))
        self.llfVal = self.LLF(hazard, betas)      # log likelihood value
        self.aicVal = self.AIC(hazard, betas)
        self.bicVal = self.BIC(hazard, betas)
        self.mvfList = self.MVF_all(hazard, omega, betas)
        
        logging.info("MVF values: {0}".format(self.mvfList))

        self.sseVal = self.SSE(self.mvfList, self.cumulativeFailures)

        self.intensityList = self.intensityFit(self.mvfList)
        logging.info("Intensity values: {0}".format(self.intensityList))