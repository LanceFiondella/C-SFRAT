from abc import ABC, abstractmethod, abstractproperty

import logging as log

import time   # for testing

import numpy as np
import sympy as sym
from sympy import symbols, diff, exp, lambdify, DeferredVector, factorial, Symbol, Idx, IndexedBase
import scipy.optimize
from scipy.special import factorial as npfactorial

import symengine

# import models   # maybe??

from core.bat import search


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
        bicVal: Bayesian information criterion value (float, used as
            goodness-of-fit measure.
        sseVal: Sum of sqaures error (float), used as goodness-of-fit measure.
        mvfList: List of results from the mean value function (float). Values
            that the model fit to the cumulative data.
        intensityList: List of values (float) that the model fit to the
            intensity data.
        config: ConfigParser object containing information about which model
            functions are implemented.
    """

    # lambdaFunctionAll = None
    maxCovariates = None

    def __init__(self, *args, **kwargs):
        """Initializes Model class

        Keyword Args:
            data: Pandas dataframe with all required columns
            metricNames: list of selected metric names
        """
        self.data = kwargs["data"]                  # dataframe
        self.metricNames = kwargs["metricNames"]    # selected metric names (strings)
        self.config = kwargs['config']
        # self.t = self.data.iloc[:, 0].values            # failure times, from first column of dataframe
        # self.failures = self.data.iloc[:, 1].values     # number of failures, from second column of dataframe
        self.t = self.data["T"].values     # failure times
        self.failures = self.data["FC"].values     # number of failures
        self.n = len(self.failures)                     # number of discrete time segments
        self.cumulativeFailures = self.data["CFC"].values
        self.totalFailures = self.cumulativeFailures[-1]
        # list of arrays or array of arrays?
        self.covariateData = [self.data[name].values for name in self.metricNames]
        self.numCovariates = len(self.covariateData)
        # self.maxCovariates = self.data.numCovariates    # total number of covariates from imported data
                                                        # data object not passed, just dataframe (which
                                                        # doesn't have numCovariates as an attribute
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
    # def LFFspecified(self):
    #     """
        
    #     """
    #     return False

    # @property
    # @abstractmethod
    # def dLFFspecified(self):
    #     """
        
    #     """
    #     return False

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
    def hazardFunction(self):
        pass

    def setupMetricString(self):
        """Creates string of metric names separated by commas"""
        if (self.metricNames == []):
            # self.metricString = "None"
            self.metricString = "None"
        else:
            self.metricString = ", ".join(self.metricNames)
        self.combinationName = f"{self.shortName} ({self.metricString})"

    def symAll(self):
        """Creates symbolic LLF for model with all metrics, and differentiates

        Called from MainWindow, when new file is imported.

        Returns:
            Lambda function implementation of the differentiated log-likelihood
            function.
        """
        t1_start = time.process_time()
        Model.maxCovariates = self.numCovariates
        f, x = self.LLF_sym(self.hazardFunction)    # pass hazard rate function
        # bh = np.array([diff(f, x[i]) for i in range(self.numCovariates + 1)])

        bh = np.array([symengine.diff(f, x[i]) for i in range(self.numCovariates + 1)])

        f = self.convertSym(x, bh, "numpy")
        t1_stop = time.process_time()
        log.info("time to convert symbolic function: %s", t1_stop - t1_start)

        return f

    def LLF_sym_old(self, hazard):
        """Log-likelihood function used for symbolic calculations.

        Symbolic variables used to allow for symbolic differentiation with
        respect to b/betas.

        Args:
            hazard: The hazard function of the implemented model.

        Returns:
            A tuple containing the symbolic log-likelihood function as the
            first element, and a vector of symbolic variables as the second
            element.
        """
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
        # x = b, b1, b2, b2 = symengine.symbols('b b1 b2 b3')
        x = symengine.symbols(f'x:{self.numCovariates + 1}')
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
        secondTerm = sum(self.failures)*symengine.log(sum(self.failures)/sum(prodlist))
        logTerm = [] #Verified
        for i in range(self.n):
            logTerm.append(self.failures[i]*symengine.log(prodlist[i]))
        thirdTerm = sum(logTerm)
        factTerm = [] #Verified
        for i in range(self.n):
            factTerm.append(symengine.log(factorial(self.failures[i])))
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
        """Converts the symbolic function to a lambda function

        Args:
            
        Returns:

        """
        # return lambdify(x, bh, target)
        return symengine.lambdify(x, bh, backend='lambda')

    def runEstimation(self):
        initial = self.initialEstimates()

        # need class of specific model being used, lambda function stored as class variable

        # only use lambda function if dLLF not defined
        if self.config[self.__class__.__name__]['dLLF'].lower() != 'yes':
            # ex. (max covariates = 3) for 3 covariates, zero_array should be length 0
            # for no covariates, zero_array should be length 3
            numZeros = Model.maxCovariates - self.numCovariates
            zero_array = np.zeros(numZeros)   # create empty array, size of num covariates


            # create new lambda function that calls lambda function for all covariates
            # for no covariates, concatenating array a with zero element array produces a

            initial = np.concatenate((initial, zero_array), axis=0)
            log.info("Initial estimates: %s", initial)

            fd = self.__class__.lambdaFunctionAll

        else:
            # use dLLF function defined by user
            fd = self.dLLF_dict[self.numCovariates]

        optimize_start = time.process_time()    # record time

        sol = self.optimizeSolution(fd, initial)


        # search_space = [[0.0, 1.0] for i in range(4)]
        # population = [self.initialEstimates() for i in range(6)]


        # print(fd(initial))

        # sol_bat = search(fd, search_space, max_generations=6, population=population,
        #    freq_min=0.021768, freq_max=0.917212, alpha=0.825154, gamma=0.82362)


        # print(sol_bat)
        # print(sol_bat[0])
        # print("plug bat into fd", fd(sol_bat[0]))
        # sol = self.optimizeSolution(fd, np.array(sol_bat[0]))
        # print(sol)


        optimize_stop = time.process_time()
        log.info("Optimization time: %s", optimize_stop - optimize_start)
        log.info("Optimized solution: %s", sol)

        self.b = sol[0]
        self.betas = sol[1:]
        # hazard = self.calcHazard(self.b, self.n)

        hazard = [self.hazardFunction(i, self.b) for i in range(self.n)]
        self.hazard = hazard    # for MVF prediction, don't want to calculate again
        self.modelFitting(hazard, self.betas)

    def runEstimation_new(self):
        f, x = self.LLF_sym(self.hazardFunction)    # pass hazard rate function
        bh = np.array([symengine.diff(f, x[i]) for i in range(self.numCovariates + 1)])
        fd = self.convertSym(x, bh, "numpy")
        log.info("Symbolic equation converted.")

        initial = self.initialEstimates()

        optimize_start = time.process_time()    # record time

        sol = self.optimizeSolution(fd, initial)

        optimize_stop = time.process_time()
        log.info("Optimization time: %s", optimize_stop - optimize_start)
        log.info("Optimized solution: %s", sol)

        self.b = sol[0]
        self.betas = sol[1:]
        # hazard = self.calcHazard(self.b, self.n)

        hazard = [self.hazardFunction(i, self.b) for i in range(self.n)]
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

        solution = scipy.optimize.fsolve(fd, x0=B)

        # try:
        #     log.info("Using broyden1")
        #     solution = scipy.optimize.broyden1(fd, xin=B, iter=1000)
        # except scipy.optimize.nonlin.NoConvergence:
        #     log.info("Using fsolve")
        #     solution = scipy.optimize.fsolve(fd, x0=B)
        # except:
        #     log.info("Could Not Converge")
        #     solution = [0 for i in range(self.numCovariates + 1)]


        #solution = scipy.optimize.broyden2(fd, xin=B)          #Does not work (Seems to work well until the 3 covariates then crashes)
        #solution = scipy.optimize.anderson(fd, xin=B)          #Works for DW2 - DS1  - EstB{0.998, 0.999} Does not work for DS2
        #solution = scipy.optimize.excitingmixing(fd, xin=B)    #Does not work

        #solution = scipy.optimize.newton_krylov(fd, xin=B)     #Does not work

        #solution = scipy.optimize.linearmixing(fd, xin=B)      #Does not work
        #solution = scipy.optimize.diagbroyden(fd, xin=B)       #Does not Work

        #solution = scipy.optimize.root(fd, x0=B, method='hybr')
        #solution = scipy.optimize.fsolve(fd, x0=B)
        log.info("MLEs solved.")
        return solution

    def modelFitting(self, hazard, betas):
        self.omega = self.calcOmega(hazard, betas)
        log.info("Calculated omega: %s", self.omega)

        # check if user implemented their own LLF
        if self.config[self.__class__.__name__]['LLF'].lower() != 'yes':
            # additional LLF function not implemented,
            # calculate LLF value using dynamic function
            self.llfVal = self.LLF(hazard, betas)      # log likelihood value
        else:
            # user implemented LLF
            # need to choose LLF for specified number of covariates
            self.llfVal = self.LLF_dict[self.numCovariates](hazard, betas)
        log.info("Calculated log-likelihood value: %s", self.llfVal)
        self.aicVal = self.AIC(hazard, betas)
        log.info("Calculated AIC: %s", self.aicVal)
        self.bicVal = self.BIC(hazard, betas)
        log.info("Calculated BIC: %s", self.bicVal)
        self.mvfList = self.MVF_all(hazard, self.omega, betas)

        # temporary
        # if (np.isnan(self.llfVal) or np.isinf(self.llfVal)):
        #     self.converged = False
        # else:
        #     self.converged = True

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

        # pass betas? or use class attribute (self.betas)?

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
        p = self.calcP(betas)
        return 2 * p - 2 * self.llfVal
        # return 2 * p - np.multiply(2, self.LLF(h, betas))
        # return 2 * 5 - 2 * -28.4042

    def BIC(self, h, betas):
        # +2 variables for any other algorithm
        p = self.calcP(betas)
        return p * np.log(self.n) - 2 * self.llfVal
        # return p * np.log(self.n) - 2 * self.LLF(h, betas)
        # return 5 * np.log(self.n) - 2 * -28.4042

    def calcP(self, betas):
        # number of covariates + number of hazard rate parameters + 1 (omega)
        return len(betas) + 1

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

        newHazard = [self.hazardFunction(i, self.b) for i in range(self.n, total_points)]  # calculate new values for hazard function
        hazard = self.hazard + newHazard

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
            covData[j].append(x[j])  # append a single variable (x[j]) to the end of each vector of covariate data

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
