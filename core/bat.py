import numpy as np
import numpy.random as random
import math

def objective_function(u):
    """Objective function used for testing. 3D sphere function.
    Keyword arguments:
    u (list) -- (x, y) values
    Returns:
    float -- value of objective function evaluated at (x, y)
    """
    #z = (1 - u[0])**2 + 100 * (u[1] - u[0]**2)**2 + (1 - u[2])**2   # 3D Rosenbrock function
    z = u[0]**2 + u[1]**2
    return z

def average_loudness(bats, pop_size):
    """Calculates the average loudness of all bats.
    Keyword arguments:
    bats (list) -- entire bat population, list elements are dictionaries representing individual bats
    pop_size (int) -- total number of bats in the population
    Returns:
    float -- average loudness of bats
    """
    loudness = [b["loudness"] for b in bats]
    return sum(loudness)/len(loudness)

def calc_pulse_rate(initial_rate, gamma, iteration):
    """Calculates the pulse rate using formula (6) from Yang's initial bat algorithm paper.
    Keyword arguments:
    initial_rate (float) -- initial pulse rate for bat
    gamma (float) -- pulse rate control parameter
    iteration (int) -- current number of main loop iterations
    Returns:
    float -- pulse rate
    """
    return initial_rate * (1 - math.exp(-1 * gamma * iteration))

def position_within_bounds(pos, search_space):
    """Ensures the new position is within bounds of the search space.
    Keyword arguments:
    pos (float) -- new created position
    search_space (list) -- search space of problem
    dimension (int) -- current dimension of loop in search function
    Returns:
    float -- position within search space
    """
    new = np.copy(pos)
    for dimension, value in enumerate(new):
        if value < search_space[dimension][0]:
            new[dimension] = search_space[dimension][0]
        elif value > search_space[dimension][1]:
            new[dimension] = search_space[dimension][1]
    return new

def local_search(bats, population_count, problem_size, new_position, best, search_space):
    avg_loudness = average_loudness(bats, population_count)     # used to calculate new position
    npos = best["position"] + np.array([random.uniform(-1, 1)*avg_loudness for i in range(problem_size)])
    return position_within_bounds(npos, search_space)

def global_search(bats, problem_size, new_position, best, search_space, i):
    bats[i]["velocity"] =   bats[i]["velocity"] + \
                            (bats[i]["position"] - best["position"]) * bats[i]["frequency"]
    pos = bats[i]["position"] + bats[i]["velocity"]
    return position_within_bounds(pos, search_space)

def init_passed_population(population, pop_size, problem_size, freq_min, freq_max, objective):
    """Initializes a bat population based on an existing population
    Keyword arguments:
    population (list) -- list of positions of all members of population
    pop_size (int) -- total number of bats in the population
    problem_size (int) -- dimension of problem (number of independent variables)
    freq_min (float) -- specified minumum bat frequency
    freq_max (float) -- specified maximum bat frequency
    objective (function) -- objective function used to evaluate population fitness
    Returns:
    list -- bat population, list elements are dictionaries representing individual bats
    """
    pop = [{"position": np.array(population[i]),
             "velocity": np.array([0]*problem_size),
             "frequency": random.uniform(freq_min, freq_max),
             "init_pulse_rate": random.random(),
             "pulse_rate": 0.0,
             "loudness": random.uniform(1.0, 2.0)   # "can typically be [1, 2]"
            } for i in range(pop_size)]
    for i in range(pop_size):
        pop[i]["fitness"] = objective(pop[i]["position"])
    return pop

def search(objective, search_space, max_generations, population,
           freq_min=0.0, freq_max=1.0, alpha=0.9, gamma=0.9):
    """Performs bat algorithm search for a global minimum of passed objective function.
    Keyword arguments:
    objective (function) -- objective function to be minimized
    search_space (list) -- bounds for each dimension of the search space
    max_generations (int) -- maximum number of generations (iterations) the main loop will run for
    population (list) -- list of positions of all initial members of population
    population_count (int) -- total number of bats in the population
    freq_range (list) -- minumum [0] and maximum [1] bat frequencies
    alpha (float) -- constant (0, 1) used to scale the decrease in loudness (default = 0.9)
    gamma (float) -- pulse control paramter (gamma > 0) (default = 0.9)
    Returns:
    list -- final bat positions
    """
    problem_size = len(search_space)    # search space provides bounds for each dimension of problem,
                                        # length of this list provides number of dimensions
    # initialize bat population using passed population
    population_count = len(population)
    bats = init_passed_population(population, population_count, problem_size, freq_min, freq_max, objective)
    new_position = np.array([0]*problem_size)     # list to hold created candidate solutions
    best = min(bats, key=lambda x:x["fitness"]).copy()         # store intial best bat, based on lowest fitness
    # main loop runs for specified number of iterations
    for t in range(max_generations):
        # loop over all bats in population
        for i in range(population_count):
            # generate new solutions by adjusting frequency, updating velocities and positions
            # calculate new frequency for bat, uniform random between min and max
            bats[i]["frequency"] = random.uniform(freq_min, freq_max)
            new_position = global_search(bats, problem_size, new_position, best, search_space, i)
            if (random.random() > bats[i]["pulse_rate"]):
                # generate local solution around selected best solution
                new_position = local_search(bats, population_count, problem_size, new_position, best, search_space)
            new_fitness = objective(new_position)   # evaluate fitness of new solution
            # new solution position replaces current bat if it has lower fitness
            # AND a random value [0, 1) is less than current loudness
            if (random.random() < bats[i]["loudness"] and new_fitness < bats[i]["fitness"]):
                bats[i]["position"] = new_position.copy()           # accept new solution
                bats[i]["fitness"] = new_fitness
                bats[i]["loudness"] = alpha * bats[i]["loudness"]   # update bat loudness
                bats[i]["pulse_rate"] = calc_pulse_rate(bats[i]["init_pulse_rate"], gamma, t)   # calculate pulse rate to be used in conditional
            # if new generated solution has better fitness than previous best, it becomes new best
            if (new_fitness < best["fitness"]):
                best["position"] = new_position.copy()
                best["fitness"] = new_fitness
    # return list of final position vectors
    final_positions = [bats[i]["position"] for i in range(population_count)]
    if __name__== "__main__":
        print("best =", best["position"], "fitness =", best["fitness"])    # un-comment to print out results
    return final_positions

def main():
    # initialize parameters
    pop_size = 25           # population size
    freq_min = 0.0          # minimum frequency
    freq_max = 1.0          # maximum frequency
    #freq_range = [0.0, 1.0]
    max_iterations = 100
    #tol = 0.00001           # stop tolerance (unused in this implementation)
    problem_size = 2        # dimensions of search space

    alpha = 0.9             # scaling constant for loudness
    gamma = 0.9             # pulse rate control parameter
                            # both set to 0.9, used in paper examples

    search_space = [[-5.0, 5.0], [-5.0, 5.0]]

    initial_pop = [[random.uniform(0.0, 1.0) for j in range(problem_size)] for i in range(pop_size)]

    final_positions = search(objective_function, search_space, max_iterations, initial_pop, pop_size, 
        freq_min, freq_max, alpha, gamma)

if __name__== "__main__":
    main()