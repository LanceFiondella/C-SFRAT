### imports ###
import sympy as sym
import numpy as np
import math
import scipy.optimize
import matplotlib.pyplot as plt

### SPECIFY DATA SETS ###

## DS1 data set
kVec = [1, 1, 2, 1, 8, 9, 6, 7, 4, 3, 0, 4, 1, 0, 2, 2, 3]
fVec = [4, 20, 1, 1, 32, 32, 24, 24, 24, 30, 0, 8, 8, 12, 20, 32, 24]
eVec = [0.0531, 0.0619, 0.158, 0.081, 1.046, 1.75, 2.96, 4.97, 0.42, 4.7, 0.9, 1.5, 2, 1.2, 1.2, 2.2, 7.6]
cVec = [1, 0, 0.5, 0.5, 2, 5, 4.5, 2.5, 4, 2, 0, 4, 6, 4, 6, 10, 8]

# DS2 data set
# kVec = [2, 11, 2, 4, 3, 1, 1, 2, 4, 0, 4, 1, 3, 0]
# fVec = [1.3, 17.8, 5.0, 1.5, 1.5, 3.0, 3.0, 8, 30, 9, 25, 15, 15, 2]
# eVec = [0.05, 1, 0.19, 0.41, 0.32, 0.61, 0.32, 1.83, 3.01, 1.79, 3.17, 3.4, 4.2, 1.2]
# cVec = [0.5, 2.8, 1, 0.5, 0.5, 1, 0.5, 2.5, 3.0, 3.0, 6, 4, 4, 1]

# edit this list to choose which covariates to perform estimation on
covariateData = [fVec, eVec, cVec]

# data info
numCovariates = len(covariateData)
n = len(kVec)
totalFailures = sum(kVec)
cumulativeFailures = np.cumsum(kVec)
t = [i for i in range(len(kVec))]  # for plotting

### HAZARD FUNCTIONS ###

# Geometric
def hazardFunction(i, b):
    f = b
    return f

# Negative Binomial (Order 2)
# def hazardFunction(self, i, b:
#     f = (i * b**2)/(1 + b * (i - 1))
#     return f

# Discrete Weibull (Order 2)
# def hazardFunction(self, i, b):
#     f = 1 - b**(i**2 - (i - 1)**2)
#     return f

# initial estimate ranges
coxParameterEstimateRange = [0.0, 0.1]
shapeParameterEstimateRange = [0.8, 0.99]

### ALL REQUIRED FUNCTIONS DEFINED BELOW ###

# symbolic log-likelihood function

def LLF_sym(hazard):
    # x[0] = b
    # x[1:] = beta1, beta2, ..

    x = sym.DeferredVector('x')
    second = []
    prodlist = []
    for i in range(n):
        sum1 = 1
        sum2 = 1
        TempTerm1 = 1
        for j in range(1, numCovariates + 1):
            TempTerm1 = TempTerm1 * sym.exp(covariateData[j - 1][i] * x[j])
        sum1 = 1 - ((1 - (hazard(i, x[0]))) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(1, numCovariates + 1):
                TempTerm2 = TempTerm2 * sym.exp(covariateData[j - 1][k] * x[j])
            sum2 = sum2 * ((1 - (hazard(i, x[0])))**(TempTerm2))
        second.append(sum2)
        prodlist.append(sum1 * sum2)

    firstTerm = -sum(kVec)  #Verified
    secondTerm = sum(kVec) * sym.log(sum(kVec) / sum(prodlist))
    logTerm = []    #Verified
    for i in range(n):
        logTerm.append(kVec[i] * sym.log(prodlist[i]))
    thirdTerm = sum(logTerm)
    factTerm = []   #Verified
    for i in range(n):
        factTerm.append(sym.log(sym.factorial(kVec[i])))
    fourthTerm = sum(factTerm)

    f = firstTerm + secondTerm + thirdTerm - fourthTerm
    return f, x

# initial parameter estimates

def initialEstimates():
        betasEstimate = np.random.uniform(coxParameterEstimateRange[0], coxParameterEstimateRange[1], numCovariates)
        bEstimate = np.random.uniform(shapeParameterEstimateRange[0], shapeParameterEstimateRange[1], 1)
        return np.insert(betasEstimate, 0, bEstimate)   # insert b in the 0th location of betaEstimate array

# optimization function

def optimizeSolution(fd, B):
    solution = scipy.optimize.fsolve(fd, x0=B)
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
    
    return solution

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
    for i in range(n):
        logTerm.append(kVec[i]*np.log(prodlist[i]))
    thirdTerm = sum(logTerm)
    factTerm = [] #Verified
    for i in range(n):
        factTerm.append(np.log(np.math.factorial(kVec[i])))
    fourthTerm = sum(factTerm)

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

# run symbolic calculations
f, x = LLF_sym(hazardFunction)
bh = np.array([sym.diff(f, x[i]) for i in range(numCovariates + 1)])
# f = symengine.lambdify(x, bh, backend='lambda')

f = sym.lambdify(x, bh, 'numpy')

# model fitting
initial = initialEstimates()
sol = optimizeSolution(f, initial)
b = sol[0]
betas = sol[1:]
hazard = [hazardFunction(i, b) for i in range(n)]

# goodness of fit
omega = calcOmega(hazard, betas)

llfVal = LLF(hazard, betas)      # log likelihood value
aicVal = AIC(hazard, betas)
bicVal = BIC(hazard, betas)
mvfList = MVF_all(hazard, omega, betas)

sseVal = SSE(mvfList, cumulativeFailures)
intensityList = intensityFit(mvfList)

# MVF results
print(mvfList)

# goodness-of-fit measures
print("LLF =", llfVal)
print("AIC =", aicVal)
print("BIC =", bicVal)
print("SSE =", sseVal)

# cumulative (MVF)
fig1, ax1 = plt.subplots()
ax1.step(t, cumulativeFailures, where='post')
ax1.set_xlabel("time")
ax1.set_ylabel("failures")
ax1.set_title("Cumulative view")
ax1.grid(True)

ax1.plot(t, mvfList, 'o')
# plt.show()

# intensity
fig2, ax2 = plt.subplots()
ax2.bar(t, kVec)
ax2.set_xlabel("time")
ax2.set_ylabel("failures")
ax2.set_title("Intensity view")
ax2.grid(True)

ax2.plot(t, intensityList, 'o', color='orange')
plt.show()