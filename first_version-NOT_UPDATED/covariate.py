#from __future__ import division
import random 
import math
#import pandas as pd
import numpy as np
#import os
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
#from itertools import chain
from time import clock
#from pandas import DataFrame
import scipy.optimize
import sympy as sym
from sympy import symbols, diff, exp, lambdify, DeferredVector, factorial, Symbol, Idx, IndexedBase
#import mpmath
import warnings
warnings.filterwarnings("ignore")

# main ui code
#import tool

# global variables
import global_variables as gv

#region excel
#----------------------------IMPORTING INPUT DATA FROM EXCEL----------------------------------------+

# python -m pip install -U pip setuptools #Package to read and write excel file on windows
#os.getcwd() #Get current working directory
#os.chdir("C:/Users/Folder/Name") #To change directory 
#os.listdir('.') #list all files and directories in the current folder
# xl = pd.ExcelFile('FailureTimes.xlsx') #Load spreadsheet  
#df = pd.read_csv("example.csv")  #For CSV
# print(xl.sheet_names) #Print all sheet names
# data = xl.parse('DS2') # Load a sheet into a DataFrame by name: df1
# n = len(data.kVec)

#y = data.kVec #based on column names
#x1 = data.EVec
#x2 = data.FVec
#x3 = data.CVec
#endregion excel

#--------------------------------------------------------------------------------------------#
#------------------------------------STEP 1--------------------------------------------------#
#--------------------------------------------------------------------------------------------#

#-------------------------------Initial estimates-------------------------------------------#
#Equation (40) -  This should also be made dynamic to the number of covariates considered

#region INITIAL ESTIMATES
def LLInitialPDF(x, n, num_covariates, covariate_data): 
    #Depends on the number of covariates considered
    B = []
    for i in x:
        B.append(i)
    firstTerm = n*math.log(n)
    
    Temp = []
    prodlist = []
    ftiTheta = []
    second = [1]
    for i in range(n):
        sum1 = 1
        TempTerm1 = 1
        for j in range(1, num_covariates + 1):
            TempTerm1 = TempTerm1 * covariate_data[j - 1][i] * B[j]
        sum1 = (1-((1-B[0])**exp(TempTerm1)))
        sum2=1
        for k in range(i):
            sum2 = sum2*((1-B[0])**exp(TempTerm1))
        prodlist.append(sum1*sum2)
        ftiTheta.append(math.log(prodlist[-1]))
        
    return -(firstTerm - n + sum(ftiTheta))

#endregion

#---------------------------------PSO for initial estimates-----------------------------------------------------------------------#

#region PSO INITIAL ESTIMATES
class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])
    def evaluate(self, costFunc, n, num_covariates, covariate_data):
        self.err_i=costFunc(self.position_i, n, num_covariates, covariate_data)
        if self.err_i<self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i.copy()
            self.err_best_i=self.err_i
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=0.5        # cognitive constant
        c2=0.5        # social constant  
        for i in range(0,num_dimensions):
            r1=random.random()  #\beta_1  # Uniformly distributed random numbers to achieve faster convergence
            r2=random.random()  #\beta_2             
            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]
            if self.position_i[i]<bounds[i][0]:
                self.position_i[i]=bounds[i][0]

def PSO(costFunc, x0, bounds, num_particles, maxiter, n, num_covariates, covariate_data, verbose=False):
    global num_dimensions
    num_dimensions=len(x0)
    err_best_g=-1                   # best error for group
    pos_best_g=[]                   # best position for group
    swarm=[]
    for i in range(0,num_particles):
        swarm.append(Particle(x0))
    i=0
    iternum=[]
    LLList = []
    ParamList = []
    timeList = []
    while i<maxiter:
        start=clock()
        if verbose: print(f'iter: {i:>4d}, best-solution: {err_best_g:10.6f}, parameters: {pos_best_g}')
        for j in range(0,num_particles):
            swarm[j].evaluate(costFunc, n, num_covariates, covariate_data)
            if swarm[j].err_i<err_best_g or err_best_g==-1:
                pos_best_g=list(swarm[j].position_i)
                err_best_g=float(swarm[j].err_i)
        for j in range(0,num_particles):
            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position(bounds)
        end=clock()
        timeList.append(end-start)
        i+=1
        iternum.append(i)
        LLList.append(err_best_g)
        ParamList.append(pos_best_g)
    # print('\nFINAL SOLUTION:')
    # print(f'   > {pos_best_g}')
    # print(f'   > {err_best_g}\n')
    return iternum, LLList, ParamList, timeList
#endregion

def PSO_initial(num_covariates):
    initial = np.random.uniform(0.0, 0.1, num_covariates+1)
    bounds = np.array([(0.5 * initial[i], 2 * initial[i]) for i in range(num_covariates+1)])
    return initial, bounds

def LLF_sym(n, num_covariates, covariate_data, kVec):
    #Equation (30)
    x = DeferredVector('x')
    second = []
    prodlist = []
    for i in range(n):
        sum1=1
        sum2=1
        TempTerm1 = 1
        for j in range(1, num_covariates + 1):
                TempTerm1 = TempTerm1 * exp(covariate_data[j - 1][i] * x[j])
        #print('Test: ', TempTerm1)
        sum1=1-((1-x[0]) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(1, num_covariates + 1):
                    TempTerm2 = TempTerm2 * exp(covariate_data[j - 1][k] * x[j])
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

def LLF(h, betas, covariate_data, n, kVec):
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

def calc_omega(h, betas, covariate_data, n, total_failures):
    # can clean this up to use less loops, probably
    covariate_num = len(betas)
    prodlist = []
    for i in range(n):
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
    denominator = sum(prodlist)
    numerator = total_failures
    # print("numerator =", numerator, "denominator =", denominator)

    return numerator / denominator

def AIC(h, betas, covariate_data, n, kVec):
    p = 5   # why?
    return 2 * p - np.multiply(2, LLF(h, betas, covariate_data, n, kVec))

def BIC(h, betas, covariate_data, n, kVec):
    p = 5   # why?
    return p * np.log(n) - 2 * LLF(h, betas, covariate_data, n, kVec)

def MVF(h, omega, betas, covariate_data, n):
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

def MVF_all(h, omega, betas, covariate_data, n):
    mvf_list = np.array([MVF(h, omega, betas, covariate_data, k) for k in range(n)])
    return mvf_list

def cumulative_failures(kvec):
    failures = np.cumsum(kvec)
    return failures

def SSE(fitted, actual):
    sub = np.subtract(fitted, actual)
    sse_error = np.sum(np.power(sub, 2))
    return sse_error

def initialization():
    # input data
    float_data = parse_raw_data(gv.raw_imported_data)
    gv.num_covariates = len(float_data) - 2    # first two columns are time and 

    gv.failure_times = float_data[0]        # global
    gv.kVec = float_data[1]
    gv.cov_data = float_data[2:len(float_data)]

    gv.n = len(gv.kVec)
    gv.kVec_cumulative = cumulative_failures(gv.kVec)  # global
    gv.total_failures = gv.kVec_cumulative[-1]

def parse_raw_data(raw_data):
    float_data = [[] for i in range(len(raw_data[0]))]
    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            try:
                float_data[j].append(float(raw_data[i][j]))
            except ValueError:
                # i += 1
                pass
    #return list(filter(None, float_data))   # remove empty elements
    return float_data

def intensity_fit(mvf_list):
    # first = np.array([mvf_list[0]])
    # difference = np.array(np.diff(mvf_list))
    # return np.concatenate(first, difference)  # want the same size as list that was input
    # print(mvf_list[0])
    difference = [mvf_list[i+1]-mvf_list[i] for i in range(len(mvf_list)-1)]
    # print(difference, type(difference))
    return [mvf_list[0]] + difference

def model_fitting(srm="geometric"):
    # select which hazard function to use
    if (srm == "nb2"):
        # negative binomial (order 2)
        gv.h = gv.nb2_hazard
    elif (srm == "dw2"):
        # discrete weibull (order 2)
        gv.h = gv.dw2_hazard
    elif (srm == "nb"):
        # negative binomial
        gv.h = gv.nb_hazard
    elif (srm == "dw"):
        # discrete weibull
        gv.h = gv.dw_hazard
    else:
        # geometric is default
        gv.h = gv.geometric_hazard

    omega = calc_omega(gv.h, gv.betas, gv.cov_data, gv.n, gv.total_failures)
    print("calculated omega =", omega)
    # omega = 41.627
    gv.llf_val = LLF(gv.h, gv.betas, gv.cov_data, gv.n, gv.kVec)      # log likelihood value
    gv.aic_val = AIC(gv.h, gv.betas, gv.cov_data, gv.n, gv.kVec)
    gv.bic_val = BIC(gv.h, gv.betas, gv.cov_data, gv.n, gv.kVec)
    gv.mvf_list = MVF_all(gv.h, omega, gv.betas, gv.cov_data, gv.n)
    
    print("MVF values:", gv.mvf_list)

    gv.sse_val = SSE(gv.mvf_list, gv.kVec_cumulative)

    gv.intensity_list = intensity_fit(gv.mvf_list)
    print("intensity values:", gv.intensity_list)

# i don't think we want to use a main function for this file,
# it should all be called in a main file i think. maybe it shouldn't be the main ui file?
def main():
    #-----------------------Define random initial values and bounds for PSO-----------------------#
    initial, bounds = PSO_initial(gv.num_covariates)

    #-----------------------Apply PSO with arbitrary value of population and iterations-----------------------#
    itert, lnL, outParam, timeiter = PSO(costFunc=LLInitialPDF, x0=initial, bounds=bounds, num_particles=5,
                                        maxiter=10, n=gv.n, num_covariates=gv.num_covariates, covariate_data=gv.cov_data, verbose=False)

    #-----------------------Define Ouput of PSO as initial parameter estimates -----------------------#
    B = np.array(outParam[-1])
    print('Initial parameter values from PSO:', outParam[-1])

    #-----------------------Define Log-likelihood function using Sympy---------------------------#
    f, x = LLF_sym(gv.n, gv.num_covariates, gv.cov_data, gv.kVec) 
    bh = np.array([diff(f, x[i]) for i in range(gv.num_covariates+1)])
    print("Log-likelihood differentiated")

    #---Convert the symbolic equations to be compatible with numpy/scipy---#
    print("Converting symbolic equation to numpy...")
    fd = lambdify(x, bh, 'numpy')
    print("Symbolic equation converted")

    #---Apply optimization method to solve for model parameters---#

    # print(B)

    sol = scipy.optimize.fsolve(fd,x0=B)      # runtime < 1 second
    print('Optimized solution:', sol)

    #print('RLL', RLLCV(sol))
    #Solution is [ 0.06936379 -0.03643947  0.04664847  0.13732605]

    # model fitting
    gv.b = sol[0]           # first element is b
    gv.betas = sol[1:]         # all remaining elements are beta values
    gv.geometric_hazard = [gv.b for i in range(gv.n)]
    print("geometric hazard:", gv.geometric_hazard[0])
    gv.nb2_hazard = [(i * np.square(gv.b))/(1 + gv.b * (i - 1)) for i in range(1, gv.n+1)]
    print("negative binomial (order 2) hazard:", gv.nb2_hazard)
    gv.dw2_hazard = [1 - np.power(gv.b, (np.square(i) - np.square(i - 1))) for i in range(1, gv.n+1)]
    print("discrete weibull (order 2) hazard:", gv.dw2_hazard)
    # gv.nb_hazard = [(i * gv.b)/(1 + gv.b * (i - 1)) for i in range(1, gv.n+1)]
    # print("negative binomial hazard:", gv.nb_hazard)
    # gv.dw_hazard = [1 - np.power(gv.b, (i - i - 1)) for i in range(1, gv.n+1)]
    # print("discrete weibull hazard:", gv.dw_hazard)

    '''
    # select which hazard function to use
    # h = geometric_hazard
    # h = nb2_hazard
    # h = dw2_hazard
    # h = nb_hazard
    gv.h = dw_hazard

    omega = calc_omega(h, betas, gv.cov_data, gv.n, total_failures)
    print("calculated omega =", omega)
    # omega = 41.627
    gv.llf_val = LLF(h, betas, gv.cov_data, gv.n, gv.kVec)      # log likelihood value
    gv.aic_val = AIC(h, betas, gv.cov_data, gv.n, gv.kVec)
    gv.bic_val = BIC(h, betas, gv.cov_data, gv.n, gv.kVec)
    gv.mvf_list = MVF_all(h, omega, betas, gv.cov_data, gv.n)
    
    print(gv.mvf_list)

    gv.sse_val = SSE(gv.mvf_list, gv.kVec_cumulative)
    '''

if __name__ == "__main__":
    #region INPUT DATA
    #-------------------------------INPUT DATA-------------------------------------------#

    #DS2
    kVec = np.array([2, 11, 2, 4, 3, 1, 1, 2, 4, 0, 4, 1, 3, 0])
    FVec = np.array([1.3, 17.8, 5.0, 1.5, 1.5, 3.0, 3.0, 8, 30, 9, 25, 15, 15, 2])
    EVec = np.array([0.05, 1, 0.19, 0.41, 0.32, 0.61, 0.32, 1.83, 3.01, 1.79, 3.17,3.4, 4.2, 1.2])
    CVec = np.array([0.5, 2.8, 1, 0.5, 0.5, 1, 0.5, 2.5, 3.0, 3.0, 6, 4, 4, 1])

    #region LIST OF LISTS OF COVARITES
    data = np.array([FVec, EVec, CVec])
    #endregion

    #region NUMBER OF COVARITES
    num_covariates = len(data)
    #endregion

    n=len(kVec)
    kVecCum = cumulative_failures(kVec)
    failure_times = np.array([i + 1 for i in range(n)])
    #endregion
    
    # input data
    # float_data = parse_raw_data(gv.raw_imported_data)
    # num_covariates = len(float_data) - 2    # first two columns are time and 

    # gv.failure_times = float_data[0]        # global
    # kVec = float_data[1]
    # cov_data = float_data[2:len(float_data)]
    cov_data = data
    kVec_cumulative = cumulative_failures(kVec)  # global
    total_failures = kVec_cumulative[-1]

    #-----------------------Define random initial values and bounds for PSO-----------------------#
    initial, bounds = PSO_initial(num_covariates)

    #-----------------------Apply PSO with arbitrary value of population and iterations-----------------------#
    itert, lnL, outParam, timeiter = PSO(costFunc=LLInitialPDF, x0=initial, bounds=bounds, num_particles=10,
                                        maxiter=20, n=n, num_covariates=num_covariates, covariate_data=cov_data, verbose=False)

    #-----------------------Define Ouput of PSO as initial parameter estimates -----------------------#
    B = np.array(outParam[-1])
    print('Initial parameter values from PSO:', outParam[-1])

    #-----------------------Define Log-likelihood function using Sympy---------------------------#
    f, x = LLF_sym(n, num_covariates, cov_data, kVec) 
    bh = np.array([diff(f, x[i]) for i in range(num_covariates+1)])
    print("Log-likelihood differentiated")

    #---Convert the symbolic equations to be compatible with numpy/scipy---#
    print("Converting symbolic equation to numpy...")
    fd = lambdify(x, bh, 'numpy')
    print("Symbolic equation converted")
    print(sym.python(fd))

    #---Apply optimization method to solve for model parameters---#

    print(B)

    sol = scipy.optimize.fsolve(fd,x0=B)      # runtime < 1 second
    print('Optimized solution:', sol)

    #print('RLL', RLLCV(sol))
    #Solution is [ 0.06936379 -0.03643947  0.04664847  0.13732605]

    # model fitting
    geometric_hazard = sol[0]   # first element is b
    betas = sol[1:]             # all remaining elements are beta values, number of covariates
                                # beta values are not in order in sol
    # betas = [sol[2], sol[1], sol[3]]

    omega = calc_omega(geometric_hazard, betas, cov_data, n, total_failures)
    print("calculated omega =", omega)
    # omega = 41.627
    print("old omega = 41.627")
    llf_val = LLF(geometric_hazard, betas, cov_data, n, kVec)      # log likelihood value
    aic_val = AIC(geometric_hazard, betas, cov_data, n, kVec)
    bic_val = BIC(geometric_hazard, betas, cov_data, n, kVec)
    mvf_list = MVF_all(geometric_hazard, omega, betas, cov_data, n)
    
    # print(gv.mvf_list)

    sse_val = SSE(mvf_list, kVec_cumulative)

    print("LLF:", llf_val)
    print("AIC:", aic_val)
    print("BIC:", bic_val)
    print("SSE:", sse_val)
    print("MVF list:", mvf_list)


#region IGNORE FOR NOW
############IGNORE FROM HERE###########################


# # For visualization


# writer = pd.ExcelWriter('NM.xlsx', engine='xlsxwriter')
# for pIndex in range(1,11):
#     LLavg=[] # Final value computed for individual particle size
#     Timeavg=[] #Total time to complete computation for different particle size
#     pSize = []
#     iSize = []
#     LLavgTemp = []
#     TimeavgTemp = []
#     bMLEavg = []
#     b1MLEavg = []
#     b2MLEavg = []
#     b3MLEavg = []
#     NMTimeAvg = []
#     NMLLAvg = []
#     NMbMLEAvg = []
#     NMb1MLEAvg = []
#     NMb2MLEAvg = []
#     NMb3MLEAvg = []
    
#     bMLEInit = []
#     b1MLEInit = []
#     b2MLEInit = []
#     b3MLEInit = []
    
#     df = []
#     iterIndex = 1
#     # for index in range(0,(2**(10-pIndex+1))):
#     for index in range(1,3):
#         LLtemp=[]
#         Timetemp=[]
#         LLtemp100runs = []
#         bMLE100runs = []
#         cMLE100runs = []
#         dfTemp = []
#         NMTimeTemp = []
#         NMbMLETemp = []
#         NMb1MLETemp = []
#         NMb2MLETemp = []
#         NMb3MLETemp = []
#         NMLLTemp = []
        
        
#         for i in range(0,1): #Number of runs #10
#             starttime = clock()
            
#             b0=random.uniform(0,0.1)
#             b10 = random.uniform(0,0.1)
#             b20 = random.uniform(0,0.1)
#             b30 = random.uniform(0,0.1)
#             initial=[b0,b10,b20,b30]    
#             bounds=[(0.5*b0,2*b0),(0.5*b10,2*b10),(0.5*b20,2*b20),(0.5*b30,2*b30)]
            
#             itert,lnL,outParam, timeiter= PSO(costFunc=LLInitialPDF,x0=initial, bounds=bounds, num_particles=20, maxiter=20, verbose=False)

#             sol=scipy.optimize.fsolve(fd,x0=(outParam[-1][0],outParam[-1][1],outParam[-1][2],outParam[-1][3]))

#             NMTimeTemp.append(clock()-starttime)

#             NMbMLETemp.append(sol[0])
#             NMb1MLETemp.append(sol[1])
#             NMb2MLETemp.append(sol[2])
#             NMb3MLETemp.append(sol[3])
#             NMLLTemp.append(RLLCV(sol))
            
#             bMLEInit.append(outParam[-1][0])
#             b1MLEInit.append(outParam[-1][1])
#             b2MLEInit.append(outParam[-1][2])
#             b3MLEInit.append(outParam[-1][3])

#         df.append(100*(sum([1 for x in NMLLTemp if math.isnan(float(x)) is False]))/10)   

#         pSize.append(pIndex)  
#         iSize.append(index)

#         NMTimeAvg.append(np.mean(NMTimeTemp))
#         NMLLAvg.append(np.mean(NMLLTemp))
#         NMbMLEAvg.append(np.mean(NMbMLETemp))
#         NMb1MLEAvg.append(np.mean(NMb1MLETemp))
#         NMb2MLEAvg.append(np.mean(NMb2MLETemp))
#         NMb3MLEAvg.append(np.mean(NMb3MLETemp))

#         bMLEavg.append(np.mean(bMLEInit))
#         b1MLEavg.append(np.mean(b1MLEInit))
#         b2MLEavg.append(np.mean(b2MLEInit))
#         b3MLEavg.append(np.mean(b3MLEInit))

#         index=index+1
#         # dfTemp = minus4[minus4['System'] == "System " + str(pIndex)]
#     dfTemp=pd.DataFrame({'Particle size': pSize, 'Iterations': iSize,'PSO_bMLE':bMLEavg,'PSO_b1MLE':b1MLEavg,'PSO_b2MLE':b2MLEavg,'PSO_b3MLE':b3MLEavg,'NMbMLE': NMbMLEAvg, 'NMb1MLE': NMb1MLEAvg , 'NMb2MLE': NMb2MLEAvg , 'NMb3MLE': NMb3MLEAvg , 'NM_LL': NMLLAvg, 'Convergence Rate': df,'NM_Time': NMTimeAvg})
#     dfTemp.to_excel(writer, sheet_name='sheet{}'.format(pIndex), index=False) #To read multiple dataframes into an excel file
#     # df.append(dfTemp(axis=0))
#     # print(df)
#     pIndex=pIndex+1
# writer.save()
#endregion