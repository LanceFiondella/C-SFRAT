from abc import ABC, abstractmethod, abstractproperty

import logging as log

import time   # for testing

import numpy as np
import scipy.optimize
from scipy.special import factorial as npfactorial

import symengine

import math


class Model(ABC):
    """Generic model class, contains functions used by all models.

    Model is an abstract base class that is never instantiated. Instead, the
    class is inherited by all implemented models, so they are able to use all
    attributes and methods of Model. Child classes are instantiated for a
    model with a specific combination of metrics.
    Model includes abstract properties and methods that need to be implemented
    by all classes that inherit from it.

    Attributes:
        data: Pandas dataframe containing the imported data for the current
            sheet.
        metricNames: List of covariate metric names as strings.
        t: Numpy array containing all failure times (T).
        failures: Numpy array containing failure counts (FC) as integers at
            each time (T).
        n: Total number of discrete time segments (int).
        cumulativeFailures: Numpy array containing cumulative failure counts
            (CFC) as integers at each time (T).
        totalFailures: Total number of failures contained in the data (int).
        covariateData: List of numpy arrays containing the data for each
            covariate metric to be used in calculations.
        numCovariates: The number of covariates to be used in calculations
            (int).
        converged: Boolean indicating if the model converged or not.
        metricString: A string containing all metric names separated by commas.
        combinationName:
        b:
        betas:
        hazard: List of the results of the hazard function as floats at each
            time.
        omega: 
        llfVal: Log-likelihood value (float), used as goodness-of-fit measure.
        aicVal: Akaike information criterion value (float), used as
            goodness-of-fit measure.
        bicVal: Bayesian information criterion value (float), used as
            goodness-of-fit measure.
        sseVal: Sum of sqaures error (float), used as goodness-of-fit measure.
        mvfList: List of results from the mean value function (float). Values
            that the model fit to the cumulative data.
        intensityList: List of values (float) that the model fit to the
            intensity data.
    """

    maxCovariates = None

    def __init__(self, *args, **kwargs):
        """Initializes Model class

        Keyword Args:
            data: Pandas dataframe with all required columns
            metricNames: list of selected metric names
        """
        self.data = kwargs["data"]                  # dataframe
        self.metricNames = kwargs["metricNames"]    # selected metric names (strings)
        self.t = self.data["T"].values     # failure times
        self.failures = self.data["FC"].values     # number of failures
        self.n = len(self.failures)                     # number of discrete time segments
        self.cumulativeFailures = self.data["CFC"].values
        self.totalFailures = self.cumulativeFailures[-1]
        self.covariateData = np.array([self.data[name].values for name in self.metricNames])
        self.numCovariates = len(self.covariateData)
        self.psseVal = None
        self.numParameters = len(self.parameterEstimates)
        self.numSymbols = self.numCovariates + self.numParameters
        self.converged = False
        self.setupMetricString()

        # logging
        log.info("---------- %s (%s) ----------", self.name, self.metricNames)
        log.debug("Failure times: %s", self.t)
        log.debug("Number of time segments: %d", self.n)
        log.debug("Failures: %s", self.failures)
        log.debug("Cumulative failures: %s", self.cumulativeFailures)
        log.debug("Total failures: %d", self.totalFailures)
        log.info("Number of covariates: %d", self.numCovariates)

    def __str__(self):
        modelString = "{0} model with {1} covariates".format(self.name, self.metricString)
        return modelString

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
    def beta0(self):
        """
        Define Cox parameter estimate range for root finding initial values
        """
        return 0.01

    @property
    @abstractmethod
    def parameterEstimates(self):
        """
        Define shape parameter estimate range for root finding initial values
        """
        return (0.1,)

    ##################################################
    # Methods that must be implemented by all models #
    ##################################################

    @abstractmethod
    def hazardSymbolic(self):
        pass

    @abstractmethod
    def hazardNumerical(self):
        pass

    ##################################################

    def setupMetricString(self):
        """Creates string of metric names separated by commas"""
        if (self.metricNames == []):
            # self.metricString = "None"
            self.metricString = "None"
        else:
            self.metricString = ", ".join(self.metricNames)
        # self.combinationName = f"{self.shortName} ({self.metricString})"
        self.combinationName = "{0} ({1})".format(self.shortName, self.metricString)

    def LLF_sym(self, hazard, covariate_data):
        # x = b, b1, b2, b2 = symengine.symbols('b b1 b2 b3')

        x = symengine.symbols(f'x:{self.numSymbols}')
        second = []
        prodlist = []
        for i in range(self.n):
            sum1 = 1
            sum2 = 1
            TempTerm1 = 1
            for j in range(self.numParameters, self.numSymbols):
                TempTerm1 = TempTerm1 * symengine.exp(covariate_data[j - self.numParameters][i] * x[j])
            sum1 = 1 - ((1 - (hazard(i + 1, x[:self.numParameters]))) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(self.numParameters, self.numSymbols):
                    TempTerm2 = TempTerm2 * symengine.exp(covariate_data[j - self.numParameters][k] * x[j])
                sum2 = sum2 * ((1 - (hazard(i + 1, x[:self.numParameters])))**(TempTerm2))
            second.append(sum2)
            prodlist.append(sum1 * sum2)

        firstTerm = -sum(self.failures)  #Verified
        secondTerm = sum(self.failures) * symengine.log(sum(self.failures) / sum(prodlist))
        logTerm = []  #Verified
        for i in range(self.n):
            logTerm.append(self.failures[i] * symengine.log(prodlist[i]))
        thirdTerm = sum(logTerm)
        factTerm = []  #Verified
        for i in range(self.n):
            factTerm.append(symengine.log(math.factorial(self.failures[i])))
        fourthTerm = sum(factTerm)

        f = firstTerm + secondTerm + thirdTerm - fourthTerm
        return f, x

    def RLL(self, x, covariate_data):
        # want everything to be array of length n
        cov_data = np.array(covariate_data)

        # gives array with dimensions numCovariates x n, just want n
        exponent_all = np.array([cov_data[i] * x[i + self.numParameters] for i in range(self.numCovariates)])

        # sum over numCovariates axis to get 1 x n array
        exponent_array = np.exp(np.sum(exponent_all, axis=0))

        h = np.array([self.hazardNumerical(i + 1, x[:self.numParameters]) for i in range(self.n)])

        one_minus_hazard = (1 - h)
        one_minus_h_i = np.power(one_minus_hazard, exponent_array)

        one_minus_h_k = np.zeros(self.n)
        for i in range(self.n):
            k_term = np.array([one_minus_hazard[i] for k in range(i)])
            
            # exponent array is just 1 for 0 covariate case, cannot index
            # have separate case for 0 covariates
            if self.numCovariates == 0:
                one_minus_h_k[i] = np.prod(np.array([one_minus_hazard[i]] * len(k_term)))
            else:
                exp_term = np.power((one_minus_hazard[i]), exponent_array[:][:len(k_term)])
                one_minus_h_k[i] = np.prod(exp_term)

        failure_sum = np.sum(self.failures)
        product_array = (1.0 - (one_minus_h_i)) * one_minus_h_k

        first_term = -failure_sum

        second_num = failure_sum
        second_denom = np.sum(product_array)

        second_term = failure_sum * np.log(second_num / second_denom)

        third_term = np.sum(np.log(product_array) * np.array(self.failures))

        fourth_term = np.sum(np.log(npfactorial(self.failures)))

        f = first_term + second_term + third_term - fourth_term
        return f

    def RLL_minimize(self, x, covariate_data):
        return -self.RLL(x, covariate_data)

    def convertSym(self, x, bh, target):
        """Converts the symbolic function to a lambda function

        Args:
            
        Returns:

        """
        return symengine.lambdify(x, bh, backend='lambda')

    def runEstimation(self, covariate_data):
        # need class of specific model being used, lambda function stored as class variable

        # ex. (max covariates = 3) for 3 covariates, zero_array should be length 0
        # for no covariates, zero_array should be length 3
        # numZeros = Model.maxCovariates - self.numCovariates
        # zero_array = np.zeros(numZeros)   # create empty array, size of num covariates


        # create new lambda function that calls lambda function for all covariates
        # for no covariates, concatenating array a with zero element array
        optimize_start = time.process_time()    # record time
        initial = self.initialEstimates()

        log.info("Initial estimates: %s", initial)
        f, x = self.LLF_sym(self.hazardSymbolic, covariate_data)    # pass hazard rate function

        bh = np.array([symengine.diff(f, x[i]) for i in range(self.numSymbols)])

        fd = self.convertSym(x, bh, "numpy")

        solution_object = scipy.optimize.minimize(self.RLL_minimize, x0=initial, args=(covariate_data,), method='Nelder-Mead')
        self.mle_array = self.optimizeSolution(fd, solution_object.x)
        optimize_stop = time.process_time()
        log.info("Optimization time: %s", optimize_stop - optimize_start)
        log.info("Optimized solution: %s", self.mle_array)

        self.modelParameters = self.mle_array[:self.numParameters]
        self.betas = self.mle_array[self.numParameters:]
        log.info("model parameters =", self.modelParameters)
        log.info("betas =", self.betas)

        hazard = np.array([self.hazardNumerical(i + 1, self.modelParameters) for i in range(self.n)])
        self.hazard_array = hazard    # for MVF prediction, don't want to calculate again
        self.modelFitting(hazard, self.mle_array, covariate_data)
        self.goodnessOfFit(self.mle_array, covariate_data)

    def initialEstimates(self):
        # bEstimate = [self.b0]
        parameterEstimates = list(self.parameterEstimates)
        betaEstimate = [self.beta0 for i in range(self.numCovariates)]
        return np.array(parameterEstimates + betaEstimate)

    def optimizeSolution(self, fd, B):
        log.info("Solving for MLEs...")

        sol_object = scipy.optimize.root(fd, x0=B)
        solution = sol_object.x
        self.converged = sol_object.success
        log.info("\t" + sol_object.message)
        
        return solution

    def modelFitting(self, hazard, mle, covariate_data):
        self.omega = self.calcOmega(hazard, self.betas, covariate_data)
        log.info("Calculated omega: %s", self.omega)

        self.mvf_array = self.MVF_all(mle, self.omega, hazard, covariate_data)
        log.info("MVF values: %s", self.mvf_array)
        self.intensityList = self.intensityFit(self.mvf_array)
        log.info("Intensity values: %s", self.intensityList)

    def goodnessOfFit(self, mle, covariate_data):
        self.llfVal = self.RLL(mle, covariate_data)
        log.info("Calculated log-likelihood value: %s", self.llfVal)

        p = self.calcP(mle)
        self.aicVal = self.AIC(p)
        log.info("Calculated AIC: %s", self.aicVal)
        self.bicVal = self.BIC(p)
        log.info("Calculated BIC: %s", self.bicVal)

        self.sseVal = self.SSE(self.mvf_array, self.cumulativeFailures)
        log.info("Calculated SSE: %s", self.sseVal)

    def calcOmega(self, h, betas, covariate_data):
        # can likely use fewer loops
        prodlist = []
        for i in range(self.n):
            sum1 = 1
            sum2 = 1
            TempTerm1 = 1
            for j in range(self.numCovariates):
                    TempTerm1 = TempTerm1 * np.exp(covariate_data[j][i] * betas[j])
            sum1 = 1-((1 - h[i]) ** (TempTerm1))
            for k in range(i):
                TempTerm2 = 1
                for j in range(self.numCovariates):
                        TempTerm2 = TempTerm2 * np.exp(covariate_data[j][k] * betas[j])
                sum2 = sum2*((1 - h[i])**(TempTerm2))
            prodlist.append(sum1*sum2)
        denominator = sum(prodlist)
        numerator = self.totalFailures

        return numerator / denominator

    def AIC(self, p):
        return 2 * p - 2 * self.llfVal

    def BIC(self, p):
        return p * np.log(self.n) - 2 * self.llfVal
        # return p * np.log(self.n) - 2 * self.LLF(h, betas)
        # return 5 * np.log(self.n) - 2 * -28.4042

    def calcP(self, mle):
        # number of covariates + number of hazard rate parameters + 1 (omega)
        return len(mle) + 1

    def MVF_all(self, mle, omega, hazard_array, covariate_data):
        mvf_array = np.array([self.MVF(mle, omega, hazard_array, dataPoints, covariate_data) for dataPoints in range(self.n)])
        return mvf_array

    def MVF(self, x, omega, hazard_array, stop, cov_data):
        # gives array with dimensions numCovariates x n, just want n
        # switched x[i + 1] to x[i + self.numParameters] to account for
        # more than 1 model parameter
        # ***** can probably change to just betas
        exponent_all = np.array([cov_data[i][:stop + 1] * x[i + self.numParameters] for i in range(self.numCovariates)])

        # sum over numCovariates axis to get 1 x n array
        exponent_array = np.exp(np.sum(exponent_all, axis=0))

        h = hazard_array[:stop + 1]

        one_minus_hazard = (1 - h)
        one_minus_h_i = np.power(one_minus_hazard, exponent_array)
        one_minus_h_k = np.zeros(stop + 1)
        for i in range(stop + 1):
            k_term = np.array([one_minus_hazard[i] for k in range(i)])
            if self.numCovariates == 0:
                one_minus_h_k[i] = np.prod(np.array([one_minus_hazard[i]] * len(k_term)))
            else:
                exp_term = np.power((one_minus_hazard[i]), exponent_array[:][:len(k_term)])
                one_minus_h_k[i] = np.prod(exp_term)

        product_array = (1.0 - (one_minus_h_i)) * one_minus_h_k

        result = omega * np.sum(product_array)
        return result

    def SSE(self, fitted, actual):
        sub = np.subtract(fitted, actual)
        sseError = np.sum(np.power(sub, 2))
        return sseError

    def intensityFit(self, mvf_array):
        difference = [mvf_array[i+1]-mvf_array[i] for i in range(len(mvf_array) - 1)]
        return [mvf_array[0]] + difference
