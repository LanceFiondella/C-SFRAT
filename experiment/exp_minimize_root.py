### imports ###
import symengine
import numpy as np
import math
import scipy.optimize
from scipy.special import factorial as npfactorial
import matplotlib.pyplot as plt

import csv
import time

### SPECIFY DATA SETS ###

# DS1 data set
kVec1 = [1, 1, 2, 1, 8, 9, 6, 7, 4, 3, 0, 4, 1, 0, 2, 2, 3]
fVec1 = [4, 20, 1, 1, 32, 32, 24, 24, 24, 30, 0, 8, 8, 12, 20, 32, 24]
eVec1 = [0.0531, 0.0619, 0.158, 0.081, 1.046, 1.75, 2.96, 4.97, 0.42, 4.7, 0.9, 1.5, 2, 1.2, 1.2, 2.2, 7.6]
cVec1 = [1, 0, 0.5, 0.5, 2, 5, 4.5, 2.5, 4, 2, 0, 4, 6, 4, 6, 10, 8]

# DS2 data set
kVec2 = [2, 11, 2, 4, 3, 1, 1, 2, 4, 0, 4, 1, 3, 0]
fVec2 = [1.3, 17.8, 5.0, 1.5, 1.5, 3.0, 3.0, 8, 30, 9, 25, 15, 15, 2]
eVec2 = [0.05, 1, 0.19, 0.41, 0.32, 0.61, 0.32, 1.83, 3.01, 1.79, 3.17, 3.4, 4.2, 1.2]
cVec2 = [0.5, 2.8, 1, 0.5, 0.5, 1, 0.5, 2.5, 3.0, 3.0, 6, 4, 4, 1]

### HAZARD FUNCTIONS ###

# Discrete Weibull (Order 2)
def discreteWeibull(i, b):
    f = 1 - b**(i**2 - (i - 1)**2)
    return f

# Geometric
def geometric(i, b):
    f = b
    return f

# Negative Binomial (Order 2)
def negativeBinomial(i, b):
    f = (i * b**2)/(1 + b * (i - 1))
    return f


# data info
# numCovariates = len(covariateData)
# n = len(kVec)
# totalFailures = sum(kVec)
# cumulativeFailures = np.cumsum(kVec)
# t = [i for i in range(len(kVec))]  # for plotting


### ALL REQUIRED FUNCTIONS DEFINED BELOW ###

# symbolic log-likelihood function

def LLF_sym(hazard):
    # x = b, b1, b2, b2 = symengine.symbols('b b1 b2 b3')
    x = symengine.symbols(f'x:{numCovariates + 1}')
    second = []
    prodlist = []
    for i in range(n):
        sum1 = 1
        sum2 = 1
        TempTerm1 = 1
        for j in range(1, numCovariates + 1):
            TempTerm1 = TempTerm1 * symengine.exp(covariateData[j - 1][i] * x[j])
        sum1 = 1 - ((1 - (hazard(i + 1, x[0]))) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(1, numCovariates + 1):
                TempTerm2 = TempTerm2 * symengine.exp(covariateData[j - 1][k] * x[j])
            sum2 = sum2 * ((1 - (hazard(i + 1, x[0])))**(TempTerm2))
        second.append(sum2)
        prodlist.append(sum1*sum2)

    firstTerm = -sum(kVec) #Verified
    secondTerm = sum(kVec)*symengine.log(sum(kVec)/sum(prodlist))
    logTerm = [] #Verified
    for i in range(n):
        logTerm.append(kVec[i]*symengine.log(prodlist[i]))
    thirdTerm = sum(logTerm)
    factTerm = [] #Verified
    for i in range(n):
        factTerm.append(symengine.log(math.factorial(kVec[i])))
    fourthTerm = sum(factTerm)

    f = firstTerm + secondTerm + thirdTerm - fourthTerm
    return f, x

def RLL_PSO(x):
    second = []
    prodlist = []
    for i in range(n):
        sum1 = 1
        sum2 = 1
        TempTerm1 = 1
        for j in range(1, numCovariates + 1):
            TempTerm1 = TempTerm1 * np.exp(covariateData[j - 1][i] * x[j])
        sum1 = 1 - ((1 - (hazardFunction(i + 1, x[0]))) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(1, numCovariates + 1):
                TempTerm2 = TempTerm2 * np.exp(covariateData[j - 1][k] * x[j])
            sum2 = sum2 * ((1 - (hazardFunction(i + 1, x[0])))**(TempTerm2))
        second.append(sum2)
        prodlist.append(sum1*sum2)

    firstTerm = -np.sum(kVec) #Verified

    secondTerm = np.sum(kVec)*np.log(np.sum(kVec)/np.sum(prodlist))
    logTerm = [] #Verified
    for i in range(n):
        logTerm.append(kVec[i]*np.log(prodlist[i]))
    thirdTerm = np.sum(logTerm)
    factTerm = [] #Verified
    for i in range(n):
        factTerm.append(np.log(math.factorial(kVec[i])))
    fourthTerm = np.sum(factTerm)

    f = -(firstTerm + secondTerm + thirdTerm - fourthTerm)  # negative for PSO minimization!
    return f

# initial parameter estimates

def initialEstimates():
    # betasEstimate = np.random.uniform(betaEstimateRange[0], betaEstimateRange[1], numCovariates)
    # bEstimate = np.random.uniform(bEstimateRange[0], bEstimateRange[1], 1)
    # return np.insert(betasEstimate, 0, bEstimate)   # insert b in the 0th location of betaEstimate array

    b = np.array(bEstimate)
    betas = np.array([betaEstimate for i in range(numCovariates)])
    return np.insert(betas, 0, b)

# for geometric, one covariate
def initial_B_function(beta):
    # JSS eq. (39)

    covariate_data = np.array(covariateData[0])

    # exponential_term = np.exp(covariate_data * beta)
    # sum1 = np.sum(exponential_term)
    # sum2 = np.sum([np.sum(exponential_term[:i + 1]) for i in range(n - 1)])

    # # print(exponential_term)
    # # print(sum2)
    # denominator = sum1 + sum2
    # exponent = -n / (denominator)
    # # print(exponent)
    # return 1 - np.exp(exponent)

    temp1 = []
    temp2 = []
    for i in range(n):
        # print("i =", i)
        temp1.append(math.exp(covariate_data[i] * beta))

        for k in range(i - 1):
            # print(k)
            temp2.append(math.exp(covariate_data[k] * beta))

    # print("b result =", 1 - math.exp(-n / (sum(temp1) + sum(temp2))))
    return 1 - math.exp(-n / (sum(temp1) + sum(temp2)))

def initial_beta_function(beta):
    # JSS eq. (40)
    covariate_data = np.array(covariateData[0])

    # if beta < 0:
    #     return 9999

    # sum1 = np.sum(covariate_data)
    # exponential_term = covariate_data * np.exp(covariate_data * beta)
    # sum2 = np.sum(exponential_term)
    # sum3 = np.sum([np.sum(exponential_term[:i + 1]) for i in range(n - 1)])

    # sum2plus3 = sum2 + sum3

    # log_multiplication_term = np.log(1 - initial_B_function(beta)) * sum2plus3

    # return -n / (sum1 + log_multiplication_term)


    temp1 = []
    temp2 = []
    for i in range(n):
        # print("i =", i)
        temp1.append(covariate_data[i] * math.exp(covariate_data[i] * beta))

        for k in range(i - 1):
            # print(k)
            temp2.append(covariate_data[k] * math.exp(covariate_data[k] * beta))

    # print(initial_B_function(beta))
    # print(beta)
    log_multiplication_term = math.log(1 - initial_B_function(beta)) * (sum(temp1) + sum(temp2))

    return -n / (sum(covariate_data) + log_multiplication_term) - beta

def objective_function(params):
    # print(params)
    return np.array([initial_B_function(params[0]), initial_beta_function(params[1])])

# optimization function

def optimizeSolution(fd, B):
    sol = scipy.optimize.root(fd, x0=B)

    return sol.x, sol.success
    

# calculate omega

def calcOmega(h, betas):
    prodlist = []
    for i in range(n):
        sum1 = 1
        sum2 = 1
        TempTerm1 = 1
        for j in range(numCovariates):
                TempTerm1 = TempTerm1 * np.exp(covariateData[j][i] * betas[j])
        sum1 = 1-((1 - h[i]) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(numCovariates):
                    TempTerm2 = TempTerm2 * np.exp(covariateData[j][k] * betas[j])
            sum2 = sum2*((1 - h[i])**(TempTerm2))
        prodlist.append(sum1*sum2)
    denominator = sum(prodlist)
    numerator = totalFailures

    return numerator / denominator

# non-symbolic log-likelihood function

def LLF(h, betas):
    second = []
    prodlist = []
    for i in range(n):
        sum1 = 1
        sum2 = 1
        TempTerm1 = 1
        for j in range(numCovariates):
            TempTerm1 = TempTerm1 * np.exp(covariateData[j][i] * betas[j])
        sum1 = 1 - ((1 - h[i]) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(numCovariates):
                TempTerm2 = TempTerm2 * np.exp(covariateData[j][k] * betas[j])
            sum2 = sum2*((1 - h[i])**(TempTerm2))
        second.append(sum2)
        prodlist.append(sum1*sum2)

    firstTerm = -sum(kVec) #Verified
    secondTerm = sum(kVec)*np.log(sum(kVec)/sum(prodlist))
    logTerm = [] #Verified

    # print("prodlist =", prodlist)

    for i in range(n):
        logTerm.append(kVec[i]*np.log(prodlist[i]))
        # print("---")
        # print("kVec =", kVec[i])
        # print("prodlist =", prodlist[i])
        # print("kVec times log =", kVec[i]*np.log(prodlist[i]))
    thirdTerm = sum(logTerm)
    factTerm = [] #Verified
    for i in range(n):
        factTerm.append(np.log(np.math.factorial(kVec[i])))
        # print("---")
        # print("kVec =", kVec[i])
        # print("factorial =", np.math.factorial(kVec[i]))
        # print("log of factorial =", factTerm[i])
    fourthTerm = sum(factTerm)

    # print("first term =", firstTerm)
    # print("second term =", secondTerm)
    # print("third term =", thirdTerm)
    # print("fourth term =", fourthTerm)

    return firstTerm + secondTerm + thirdTerm - fourthTerm

def RLL(x):
    second = []
    prodlist = []
    for i in range(n):
        sum1 = 1
        sum2 = 1
        TempTerm1 = 1
        for j in range(1, numCovariates + 1):
            TempTerm1 = TempTerm1 * np.exp(covariateData[j - 1][i] * x[j])
        sum1 = 1 - ((1 - (hazardFunction(i + 1, x[0]))) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(1, numCovariates + 1):
                TempTerm2 = TempTerm2 * np.exp(covariateData[j - 1][k] * x[j])
            sum2 = sum2 * ((1 - (hazardFunction(i + 1, x[0])))**(TempTerm2))
        second.append(sum2)
        prodlist.append(sum1*sum2)

    firstTerm = -np.sum(kVec) #Verified

    secondTerm = np.sum(kVec)*np.log(np.sum(kVec)/np.sum(prodlist))
    logTerm = [] #Verified
    for i in range(n):
        logTerm.append(kVec[i]*np.log(prodlist[i]))
    thirdTerm = np.sum(logTerm)
    factTerm = [] #Verified
    for i in range(n):
        factTerm.append(np.log(math.factorial(kVec[i])))
    fourthTerm = np.sum(factTerm)

    # print("correct first term =", firstTerm)
    # print("correct second term =", secondTerm)
    # print("correct third term =", thirdTerm)
    # print("correct fourth term =", fourthTerm)
    # print("correct prodlist =", prodlist)

    f = -(firstTerm + secondTerm + thirdTerm - fourthTerm)  # negative for minimization!
    return f

def calcP(betas):
    # number of covariates + number of hazard rate parameters + 1 (omega)
    return len(betas) + 1

def AIC(h, betas, llfVal):
    # +2 variables for any other algorithm
    p = calcP(betas)
    return 2 * p - 2 * llfVal

def BIC(h, betas, llfVal):
    # +2 variables for any other algorithm
    p = calcP(betas)
    return p * np.log(n) - 2 * llfVal

def MVF(h, omega, betas, stop):
    prodlist = []
    for i in range(stop + 1):
        sum1 = 1
        sum2 = 1
        TempTerm1 = 1
        for j in range(numCovariates):
            TempTerm1 = TempTerm1 * np.exp(covariateData[j][i] * betas[j])
        sum1 = 1-((1 - h[i]) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(numCovariates):
                TempTerm2 = TempTerm2 * np.exp(covariateData[j][k] * betas[j])
            sum2 = sum2 * ((1 - h[i])**(TempTerm2))
        prodlist.append(sum1 * sum2)
    return omega * sum(prodlist)

# creates the list/array of all MVF values

def MVF_all(h, omega, betas):
    mvf_array = np.array([MVF(h, omega, betas, dataPoints) for dataPoints in range(n)])
    return mvf_array

def SSE(fitted, actual):
    sub = np.subtract(fitted, actual)
    sseError = np.sum(np.power(sub, 2))
    return sseError

# calculates fitted values for intensity data

def intensityFit(mvf_array):
    difference = [mvf_array[i+1]-mvf_array[i] for i in range(len(mvf_array)-1)]
    return [mvf_array[0]] + difference


### RUN ESTIMATION AND CALCULATE GOODNESS OF FIT MEASURES ###

def experiment(iterations, dataset_string, model_string, cov_string):

    # rows = [None for r in range(iterations)]
    time_list = []

    for loop in range(iterations):

        start = time.process_time()

        # run symbolic calculations
        f, x = LLF_sym(hazardFunction)
        bh = np.array([symengine.diff(f, x[i]) for i in range(numCovariates + 1)])
        f = symengine.lambdify(x, bh, backend='lambda')

        # initial estimates; b, betas
        initial = initialEstimates()
        solution_object = scipy.optimize.minimize(RLL, x0=initial, method='Nelder-Mead', options={'maxiter': 20 * (numCovariates + 1)})
        # solution_object = scipy.optimize.minimize(RLL_new, x0=initial, method='Nelder-Mead')
        initial = solution_object.x

        # calculate MLEs
        sol, convergence = optimizeSolution(f, initial)

        # record elapsed time
        stop = time.process_time()
        elapsed_time = stop - start
        time_list.append(elapsed_time)

        b = sol[0]
        betas = sol[1:]
        hazard = [hazardFunction(i, b) for i in range(1, n + 1)]

        # model fitting
        omega = calcOmega(hazard, betas)
        llfVal = LLF(hazard, betas)      # log likelihood value
        aicVal = AIC(hazard, betas, llfVal)
        bicVal = BIC(hazard, betas, llfVal)
        mvfList = MVF_all(hazard, omega, betas)
        sseVal = SSE(mvfList, cumulativeFailures)
        intensityList = intensityFit(mvfList)

    row = [dataset_string, model_string, cov_string, initial[0], initial[1:], b, betas, np.mean(np.array(time_list)), llfVal, convergence]

    print("calculated LLF =", llfVal)
    print("mean elapsed time:", np.mean(np.array(time_list)))
    print("converged:", convergence)

    ## write to csv
    # with open(filename, 'a+', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(row)

if __name__ == "__main__":
    ##################################################
    #           CHANGE FOR EACH EXPERIMENT           #
    ##################################################

    filename = 'results/minimize_20_iterations.csv'
    num_iterations = 1

    # kVec = kVec1
    # dataset = "DS1"
    # hazardFunction = discreteWeibull
    # model = "DW"
    # covariateData = []
    # covariate_string = "-"
    # bEstimate = 0.994
    # betaEstimate = 0.01

    # numCovariates = len(covariateData)
    # n = len(kVec)
    # totalFailures = sum(kVec)
    # cumulativeFailures = np.cumsum(kVec)

    ## -------- DISCRETE WEIBULL --------
    hazardFunction = discreteWeibull
    model = "DW"
    bEstimate = 0.994
    betaEstimate = 0.01
    # -- DS1 --
    kVec = kVec1
    dataset = "DS1"
    n = len(kVec)
    totalFailures = sum(kVec)
    cumulativeFailures = np.cumsum(kVec)

    covariateData = []
    covariate_string = "-"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec1]
    covariate_string = "E"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [fVec1]
    covariate_string = "F"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [cVec1]
    covariate_string = "C"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec1, fVec1]
    covariate_string = "EF"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [fVec1, cVec1]
    covariate_string = "FC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec1, cVec1]
    covariate_string = "EC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)    

    # covariateData = [eVec1, fVec1, cVec1]
    # covariate_string = "EFC"
    # numCovariates = len(covariateData)
    # experiment(num_iterations, dataset, model, covariate_string)
    # print("******* NEW RLL (DW EFC):", RLL_new(np.array([0.99781597137126, 0.0713078819510549, 0.0361962602734462, 0.058435104600563])))

    # -- DS2 --
    kVec = kVec2
    dataset = "DS2"
    n = len(kVec)
    totalFailures = sum(kVec)
    cumulativeFailures = np.cumsum(kVec)

    covariateData = []
    covariate_string = "-"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec2]
    covariate_string = "E"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [fVec2]
    covariate_string = "F"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [cVec2]
    covariate_string = "C"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec2, fVec2]
    covariate_string = "EF"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [fVec2, cVec2]
    covariate_string = "FC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec2, cVec2]
    covariate_string = "EC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec2, fVec2, cVec2]
    covariate_string = "EFC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    ## -------- GEOMETRIC --------
    hazardFunction = geometric
    model = "GM"
    bEstimate = 0.01
    betaEstimate = 0.01
    # -- DS1 --
    kVec = kVec1
    dataset = "DS1"
    n = len(kVec)
    totalFailures = sum(kVec)
    cumulativeFailures = np.cumsum(kVec)

    covariateData = []
    covariate_string = "-"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec1]
    covariate_string = "E"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [fVec1]
    covariate_string = "F"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [cVec1]
    covariate_string = "C"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec1, fVec1]
    covariate_string = "EF"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [fVec1, cVec1]
    covariate_string = "FC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec1, cVec1]
    covariate_string = "EC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec1, fVec1, cVec1]
    covariate_string = "EFC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    # -- DS2 --
    kVec = kVec2
    dataset = "DS2"
    n = len(kVec)
    totalFailures = sum(kVec)
    cumulativeFailures = np.cumsum(kVec)

    covariateData = []
    covariate_string = "-"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec2]
    covariate_string = "E"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [fVec2]
    covariate_string = "F"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [cVec2]
    covariate_string = "C"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec2, fVec2]
    covariate_string = "EF"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [fVec2, cVec2]
    covariate_string = "FC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec2, cVec2]
    covariate_string = "EC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec2, fVec2, cVec2]
    covariate_string = "EFC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    # -------- NEGATIVE BINOMIAL --------
    hazardFunction = negativeBinomial
    model = "NB"
    bEstimate = 0.01
    betaEstimate = 0.01
    # -- DS1 --
    kVec = kVec1
    dataset = "DS1"
    n = len(kVec)
    totalFailures = sum(kVec)
    cumulativeFailures = np.cumsum(kVec)

    covariateData = []
    covariate_string = "-"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec1]
    covariate_string = "E"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [fVec1]
    covariate_string = "F"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [cVec1]
    covariate_string = "C"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec1, fVec1]
    covariate_string = "EF"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [fVec1, cVec1]
    covariate_string = "FC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec1, cVec1]
    covariate_string = "EC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec1, fVec1, cVec1]
    covariate_string = "EFC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    # -- DS2 --
    kVec = kVec2
    dataset = "DS2"
    n = len(kVec)
    totalFailures = sum(kVec)
    cumulativeFailures = np.cumsum(kVec)

    covariateData = []
    covariate_string = "-"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec2]
    covariate_string = "E"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [fVec2]
    covariate_string = "F"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [cVec2]
    covariate_string = "C"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec2, fVec2]
    covariate_string = "EF"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [fVec2, cVec2]
    covariate_string = "FC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec2, cVec2]
    covariate_string = "EC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)

    covariateData = [eVec2, fVec2, cVec2]
    covariate_string = "EFC"
    numCovariates = len(covariateData)
    experiment(num_iterations, dataset, model, covariate_string)
