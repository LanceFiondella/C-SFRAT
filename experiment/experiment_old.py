### imports ###
import symengine
import numpy as np
import math
import scipy.optimize
import matplotlib.pyplot as plt

import csv
import time

from core.optimization import PSO

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

##################################################
#           CHANGE FOR EACH EXPERIMENT           #
##################################################

# edit this list to choose which covariates to perform estimation on
# covariateData = [fVec, eVec, cVec]
# covariates = "F"

# expected_llf = -23.2909

# initial estimates
# bEstimate = 0.01
# betaEstimate = 0.01

# dataset = "DS1"
# model = "GM"

filename = 'experiment_raw/experiment4.csv'

##################################################

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
    solution, infodict, ier, mesg = scipy.optimize.fsolve(fd, x0=B, maxfev=1000, full_output=True)
    # solution = scipy.optimize.brentq(fd, -1.0, 1.0)

    # try:
    #     log.info("Using broyden1")
    #     solution = scipy.optimize.broyden1(fd, xin=B, iter=1000)
    # except scipy.optimize.nonlin.NoConvergence:
    #     log.info("Using fsolve")
    #     solution = scipy.optimize.fsolve(fd, x0=B)
    # except:
    #     log.info("Could Not Converge")
    #     solution = [0 for i in range(numCovariates + 1)]

    print("solution found?  ", ier)
    print(infodict)
    print(mesg)
    
    return solution, ier

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

def calcP(betas):
    # number of covariates + number of hazard rate parameters + 1 (omega)
    return len(betas) + 1

def AIC(h, betas):
    # +2 variables for any other algorithm
    p = calcP(betas)
    return 2 * p - 2 * llfVal

def BIC(h, betas):
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
    # iterations = 1

    rows = [None for r in range(iterations)]

    for loop in range(iterations):

        start = time.process_time()

        # run symbolic calculations
        f, x = LLF_sym(hazardFunction)
        bh = np.array([symengine.diff(f, x[i]) for i in range(numCovariates + 1)])
        f = symengine.lambdify(x, bh, backend='lambda')

        # model fitting
        initial = initialEstimates()

        bounds = [(0.5 * bEstimate, 2 * bEstimate)]
        betaBounds = [(0.5 * betaEstimate, 2 * betaEstimate) for i in range(numCovariates)]
        bounds = bounds + betaBounds

        beta0 = scipy.optimize.fsolve(initial_beta_function, x0=betaEstimate)
        # beta0 = scipy.optimize.brentq(initial_beta_function, a=0.0, b=1.0)
        # initial = scipy.optimize.fsolve(objective_function, x0=initial)
        
        b0 = initial_B_function(beta0)

        print("b0 =", b0)
        print("beta0 =", beta0)
        
        initial = np.array([b0, beta0], dtype='float64')


        # x = np.linspace(-1.0, 1.0, num=100)

        # fig, ax = plt.subplots()
        # ax.plot(x, [initial_beta_function(x[i]) for i in range(100)])


        # fig3d = plt.figure()
        # ax3d = fig3d.gca(projection='3d')
        # X, Y = np.meshgrid(x, x)
        # Z = initial


        # fig, ax = plt.subplots()
        # ax.plot(x, [initial_B_function(x[i]) for i in range(1000)])

        # plt.show()

        sol, convergence = optimizeSolution(f, outParamtmp[-1])
        # sol, convergence = optimizeSolution(f, initial)

        stop = time.process_time()

        b = sol[0]
        betas = sol[1:]
        hazard = [hazardFunction(i, b) for i in range(1, n + 1)]
        # print("hazard =", hazard)
        # print("b =", b)
        # print("betas =", betas)

        # model fitting
        omega = calcOmega(hazard, betas)

        llfVal = LLF(hazard, betas)      # log likelihood value
        aicVal = AIC(hazard, betas)
        bicVal = BIC(hazard, betas)
        mvfList = MVF_all(hazard, omega, betas)

        sseVal = SSE(mvfList, cumulativeFailures)
        intensityList = intensityFit(mvfList)

        elapsed_time = stop - start
        print("elapsed time =", elapsed_time)

        # check for convergence, converged if within 0.001 of actual llf
        if convergence == 1:
            converged = "YES"
        else:
            converged = "NO"

        # check for convergence
        # if expected_llf * 1.001 <= llfVal <= expected_llf * 0.999:
        #     converged = "YES"
        # else:
        #     converged = "NO"

        
        # print("expected LLF =", expected_llf)
        print("calculated LLF =", llfVal)
        print("converged:", converged)

        # llf_difference = expected_llf - llfVal

        # rows[loop] = [dataset, model, covariates, b, betas, elapsed_time, expected_llf, llfVal, llf_difference, converged]
        rows[loop] = [dataset_string, model_string, cov_string, b, betas, elapsed_time, llfVal, converged]

# write to csv
# fields = ["b", "betas", "time", "LLF", "converged"]

# with open(filename, 'a+', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     # csvwriter.writerow(fields)
#     csvwriter.writerow(rows[0])

# print("b =", b)
# print("betas =", betas)

# # MVF results
# print(mvfList)

# # goodness-of-fit measures
# print("LLF =", llfVal)
# print("AIC =", aicVal)
# print("BIC =", bicVal)
# print("SSE =", sseVal)

# # cumulative (MVF)
# fig1, ax1 = plt.subplots()
# ax1.step(t, cumulativeFailures, where='post')
# ax1.set_xlabel("time")
# ax1.set_ylabel("failures")
# ax1.set_title("Cumulative view")
# ax1.grid(True)

# ax1.plot(t, mvfList, 'o')
# # plt.show()

# # intensity
# fig2, ax2 = plt.subplots()
# ax2.bar(t, kVec)
# ax2.set_xlabel("time")
# ax2.set_ylabel("failures")
# ax2.set_title("Intensity view")
# ax2.grid(True)

# ax2.plot(t, intensityList, 'o', color='orange')
# plt.show()

if __name__ == "__main__":
    kVec = kVec1
    covariateData = []
    hazardFunction = geometric
    bEstimate = 0.01
    betaEstimate = 0.01
    numCovariates = len(covariateData)
    n = len(kVec)
    totalFailures = sum(kVec)
    cumulativeFailures = np.cumsum(kVec)
    experiment(100, "DS1", "DW", "-")
