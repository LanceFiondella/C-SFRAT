#from scipy.optimize import shgo

def effortAllocation(model, B, f):
    """
    # cons = ({'type': 'ineq', 'fun': lambda x:  B-x[0]-x[1]-x[2]})
    cons = ({'type': 'ineq', 'fun': lambda x:  B - sum([x[i] for i in range(model.numCovariates)])})
    # bnds = ((0, None), (0, None), (0, None))
    bnds = tuple((0, None) for i in range(model.numCovariates))

    res = shgo(model.allocationFunction, args=(f,), bounds=bnds, constraints=cons)#, n=10000, iters=4)
    # res = shgo(lambda x: -(51+ 1.5449911694401008*(1- (0.9441308828628996 ** (np.exp(0.10847739229960603*x[0]+0.027716725008716442*x[1]+0.159319065848297*x[2]))))), bounds=bnds, constraints=cons, n=10000, iters=4)
    print(res)
    print(sum(res.x))

    return res
    """
    pass