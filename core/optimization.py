import random


class Particle:
    def __init__(self, x0):
        self.position_i = []          # particle position
        self.velocity_i = []          # particle velocity
        self.pos_best_i = []          # best position individual
        self.err_best_i = -1          # best error individual
        self.err_i = -1               # error individual

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)

        # print(self.err_i)

        # if self.err_i == float("-inf"):
        #     self.err_i = float("inf")

        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i.copy()
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w = 0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1 = 0.5        # cognitive constant
        c2 = 0.5        # social constant

        for i in range(0, num_dimensions):
            r1 = random.random()  #\beta_1  # Uniformly distributed random numbers to achieve faster convergence
            r2 = random.random()  #\beta_2 

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]

#Converted PSO from class object to a function to return output recorded during PSO iterations
def PSO(costFunc, x0, bounds, num_particles, maxiter, verbose=False):
    global num_dimensions
    num_dimensions = len(x0)
    err_best_g = -1                   # best error for group
    pos_best_g = []                   # best position for group

        # establish the swarm
    swarm = []
    for i in range(0, num_particles):
        swarm.append(Particle(x0))

        # begin optimization loop
    i = 0
    iternum = []
    LLList = []
    ParamList = []
    timeList = []
    while i < maxiter:
        # start=clock()
        if verbose: print(f'iter: {i:>4d}, best-solution: {err_best_g:10.6f}, parameters: {pos_best_g}')
        # cycle through particles in swarm and evaluate fitness
        for j in range(0, num_particles):
            swarm[j].evaluate(costFunc)
            # determine if current particle is the best (globally)
            if swarm[j].err_i < err_best_g or err_best_g == -1:
                pos_best_g = list(swarm[j].position_i)
                err_best_g = float(swarm[j].err_i)
            
            # cycle through swarm and update velocities and position
        for j in range(0, num_particles):
            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position(bounds)
        # end=clock()
        # timeList.append(end-start)
        i += 1
        iternum.append(i)
        LLList.append(err_best_g)
        ParamList.append(pos_best_g)
    print('\nFINAL SOLUTION:')
    print(f'   > {pos_best_g}')
    print(f'   > {err_best_g}\n')
    return iternum, LLList, ParamList, timeList

def PSO_main(costFunc, x0, bounds, num_particles, maxiter):
    result = []
    for dimension in range(len(x0)):
        itertmp, lnLtmp, outParamtmp, timeiterTemp = PSO(costFunc, x0,
                bounds, num_particles, maxiter, dimension, verbose=False)
        result.append(outParamtmp[-1][dimension])
    return result