#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import random
import math
import pandas as pd
import numpy as np
import scipy.optimize
import os
from pandas import DataFrame
from itertools import chain
from time import clock
import mpmath
import warnings

from core.bat import search

old_settings = np.seterr(all='ignore')

#----------------------------IMPORTING INPUT DATA FROM EXCEL----------------------------------------+

kVec = np.array([2, 11, 2, 4, 3, 1, 1, 2, 4, 0, 4, 1, 3, 0])
EVec = np.array([0.05, 1, 0.19, 0.41, 0.32, 0.61, 0.32, 1.83, 3.01, 1.79, 3.17,3.4, 4.2, 1.2])
FVec = np.array([1.3, 17.8, 5.0, 1.5, 1.5, 3.0, 3.0, 8, 30, 9, 25, 15, 15, 2])
CVec = np.array([0.5, 2.8, 1, 0.5, 0.5, 1, 0.5, 2.5, 3.0, 3.0, 6, 4, 4, 1])

n=len(kVec)

#-------------------------------- COST/OBJECTIVE FUNCTION ------------------------------------------------------------+
def RLLPSO(x):
    b, b1, b2, b3 = x
    
    second = []
    prodlist = []
    for i in range(n):
        sum1=1
        sum2=1
        sum1=1-((1-b)**(np.exp(EVec[i]*b1)*np.exp(FVec[i]*b2)*np.exp(CVec[i]*b3)))
        for k in range(i):
            sum2 = sum2*((1-b)**(np.exp(EVec[k]*b1)*np.exp(FVec[k]*b2)*np.exp(CVec[k]*b3)))
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
        factTerm.append(np.log(math.factorial(kVec[i])))
    fourthTerm = sum(factTerm)
    
    return -(firstTerm + secondTerm + thirdTerm - fourthTerm)   # negative, PSO tries to minimize


def RLL(x):
    b = x[0]
    b1 = x[1]
    b2 = x[2]
    b3 = x[3]
    
    second = []
    prodlist = []
    for i in range(n):
        sum1=1
        sum2=1
        sum1=1-((1-b)**(np.exp(EVec[i]*b1)*np.exp(FVec[i]*b2)*np.exp(CVec[i]*b3)))
        for k in range(i):
            sum2 = sum2*((1-b)**(np.exp(EVec[k]*b1)*np.exp(FVec[k]*b2)*np.exp(CVec[k]*b3)))
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
        factTerm.append(np.log(math.factorial(kVec[i])))
    fourthTerm = sum(factTerm)
    
    return -(firstTerm + secondTerm + thirdTerm - fourthTerm)
    # result = -(firstTerm + secondTerm + thirdTerm - fourthTerm)

    # return [result, result, result, result]


#--- MAIN ---------------------------------------------------------------------+

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
        

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i<self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i.copy()
            self.err_best_i=self.err_i
                    
    # update new particle velocity
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

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            
            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i]<bounds[i][0]:
                self.position_i[i]=bounds[i][0]
        
#Converted PSO from class object to a function to return output recorded during PSO iterations
def PSO(costFunc, x0, bounds, num_particles, maxiter, verbose=False):
    global num_dimensions
    num_dimensions=len(x0)
    err_best_g=-1                   # best error for group
    pos_best_g=[]                   # best position for group

        # establish the swarm
    swarm=[]
    for i in range(0,num_particles):
        swarm.append(Particle(x0))

        # begin optimization loop
    i=0
    iternum=[]
    LLList = []
    ParamList = []
    timeList = []
    while i<maxiter:
        start = clock()
        if verbose: print(f'iter: {i:>4d}, best-solution: {err_best_g:10.6f}, parameters: {pos_best_g}')
        # cycle through particles in swarm and evaluate fitness
        for j in range(0,num_particles):
            swarm[j].evaluate(costFunc)
            # determine if current particle is the best (globally)
            if swarm[j].err_i<err_best_g or err_best_g==-1:
                pos_best_g=list(swarm[j].position_i)
                err_best_g=float(swarm[j].err_i)
            
            # cycle through swarm and update velocities and position
        for j in range(0, num_particles):
            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position(bounds)
        end=clock()
        timeList.append(end-start)
        i+=1
        iternum.append(i)
        LLList.append(err_best_g)
        ParamList.append(pos_best_g)
    print('\nFINAL SOLUTION:')
    print(f'   > {pos_best_g}')
    print(f'   > {err_best_g}\n')
    return iternum, LLList, ParamList, timeList

if __name__ == "__PSO__":
    main()

#---------------------------Initial estimates----------
b0=random.uniform(0,0.1)
b10 = random.uniform(0,0.1)
b20 = random.uniform(0,0.1)
b30 = random.uniform(0,0.1)

def initialEstimates():
    a = random.uniform(0.8, 0.99)    # geometric distribution variable, 0.8 - 0.99
    b = random.uniform(0, 0.1)
    c = random.uniform(0, 0.1)
    d = random.uniform(0, 0.1)

    return [a, b, c, d]


#----------------------------Run PSO ----------------------------------------------------------------------+
starttime = clock()
initial=[b0,b10,b20,b30]              
bounds=[(0.5*b0,2*b0),(0.5*b10,2*b10),(0.5*b20,2*b20),(0.5*b30,2*b30)]

#Change "num_particles=2**4, maxiter=2**3" in the below line if it is slow
itertmp,lnLtmp,outParamtmp, timeiterTemp=PSO(costFunc=RLLPSO, x0=initial, bounds=bounds, num_particles=2**5, maxiter=2**4, verbose=False)






# search_space = [[0.0001, 0.9999] for i in range(4)]
# population = [initialEstimates() for i in range(6)]


# sol = search(RLLPSO, search_space, max_generations=6, population=population,
#            freq_min=0.021768, freq_max=0.917212, alpha=0.825154, gamma=0.82362)
# print(sol)


# temp = []
# for i in range(len(sol)):
#     temp.append(RLL(sol[i]))
# best_index = temp.index(max(temp))

# solution = scipy.optimize.minimize(RLL, x0=sol[best_index])

# print(outParamtmp[-1], lnLtmp[-1]) #Use outParamtmp[-1] output as initial estimates to the fsolve or other algorithm

# bounds = [(0.0, 1.0) for i in range(4)]
# solution = scipy.optimize.minimize(RLL, method='L-BFGS-B', x0=outParamtmp[-1], bounds=bounds, options={'gtol': 1e-10, 'disp': True})
# print(solution)

# print("LLF value:", RLL(solution.x))

print(clock()-starttime)
