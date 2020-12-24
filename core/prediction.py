import numpy as np


def prediction_mvf(model, failures, covariate_data, effortDict):
    """
    effortDict: dictionary containing all prediction effort spin box widgets,
        indexed by covariate string
    """

    # print(effortDict)
    total_points = model.n + failures

    # new_array = np.zeros(self.numCovariates)
    new_array = []
    # i = 0
    for cov in model.metricNames:
        value = effortDict[cov].value()
        new_array.append(np.full(failures, value))
        # i += 1
    # print(new_array)

    if model.numCovariates == 0:
        combined_array = np.concatenate((covariate_data, np.array(new_array)))
    else:
        combined_array = np.concatenate((covariate_data, np.array(new_array)), axis=1)

    # print(combined_array)

    newHazard = np.array([model.hazardFunction(i, model.modelParameters) for i in range(model.n, total_points)])  # calculate new values for hazard function
    # hazard = self.hazard_array + newHazard
    hazard = np.concatenate((model.hazard_array, newHazard))


    ## VERIFY OMEGA VALUE, should we continue updating??

    # omega = self.calcOmega(hazard, self.betas, new_covData)
    omega = model.calcOmega(hazard, model.betas, combined_array)

    mvf_array = np.array([model.MVF(model.mle_array, omega, hazard, dataPoints, combined_array) for dataPoints in range(total_points)])
    # intensity_array = self.intensityFit(mvf_array)
    x = np.concatenate((model.t, np.arange(model.n + 1, total_points + 1)))

    # add initial point at zero if not present
    # if self.t[0] != 0:
    #     mvf_array = np.concatenate((np.zeros(1), mvf_array))
    #     intensity_array = np.concatenate((np.zeros(1), intensity_array))

    return (x, mvf_array)

def prediction_psse(model, data):
    """
    Prediction function used for PSSE. Imported covariate data is used.
    """

    total_points = data.max_interval

    full_data = data.getData()

    covariateData = np.array([full_data[name] for name in model.metricNames])

    newHazard = np.array([model.hazardFunction(i, model.modelParameters) for i in range(model.n, total_points)])  # calculate new values for hazard function
    # hazard = self.hazard_array + newHazard
    hazard = np.concatenate((model.hazard_array, newHazard))


    ## VERIFY OMEGA VALUE, should we continue updating??

    # omega = self.calcOmega(hazard, self.betas, new_covData)
    omega = model.calcOmega(hazard, model.betas, covariateData)

    mvf_array = np.array([model.MVF(model.mle_array, omega, hazard, dataPoints, covariateData) for dataPoints in range(total_points)])

    return mvf_array

def prediction_intensity(model, intensity, covariate_data, effortDict):
    # res = scipy.optimize.root_scalar(self.pred_function, x0=self.n+1, x1=self.n+3, args=(intensity, covariate_data))
    # print(res)

    #########################################

    mvf_list = model.mvf_array.tolist()

    for i in range(1, 100):
        total_points = model.n + i

        # new_array = np.zeros(self.numCovariates)
        new_array = []
        j = 0
        for cov in model.metricNames:
            value = effortDict[cov].value()
            new_array.append(np.full(i, value))
            j += 1
        print(new_array)

        if model.numCovariates == 0:
            combined_array = np.concatenate((covariate_data, np.array(new_array)))
        else:
            combined_array = np.concatenate((covariate_data, np.array(new_array)), axis=1)

        print(combined_array)

        newHazard = np.array([model.hazardFunction(j, model.modelParameters) for j in range(model.n, total_points)])  # calculate new values for hazard function
        # hazard = self.hazard_array + newHazard
        hazard = np.concatenate((model.hazard_array, newHazard))



        ## VERIFY OMEGA VALUE, should we continue updating??

        # omega = self.calcOmega(hazard, self.betas, new_covData)
        omega = model.calcOmega(hazard, model.betas, combined_array)

        # print(omega)
        # print(self.omega)


        #### IGNORE IF 0 !!!!!! ####

        mvf_list.append(model.MVF(model.mle_array, omega, hazard, total_points - 1, combined_array))
        calculated_intensity = mvf_list[-1] - mvf_list[-2]
        print("calculated intensity:", calculated_intensity)
        print("desired intensity:", intensity)
        if calculated_intensity < intensity:
            print("desired failure intensity reached in {0} intervals".format(i))
            x = np.concatenate((model.t, np.arange(model.n + 1, len(mvf_list) + 1)))
            return (x, model.intensityFit(mvf_list), i)

    print("desired failure intensity not reached within 100 intervals")
    return (model.t, model.intensityList, 0)


# def pred_function(self, n, intensity, covariate_data):
#     print("n =", n)
#     x = int(n)
#     print("x =", x)

#     zero_array = np.zeros(x - self.n)    # to append to existing covariate data
#     new_covData = [0 for i in range(self.numCovariates)]

#     newHazard = np.array([self.hazardFunction(i, self.modelParameters) for i in range(self.n, x)])  # calculate new values for hazard function
#     # hazard = self.hazard_array + newHazard
#     hazard = np.concatenate((self.hazard_array, newHazard))

#     for j in range(self.numCovariates):
#         new_covData[j] = np.append(covariate_data[j], zero_array)

#     # print(self.n)
#     # print(len(new_covData))
    
#     omega1 = self.calcOmega(hazard, self.betas, new_covData)
#     omega2 = self.calcOmega(hazard[:-1], self.betas, new_covData)

#     # print(len(hazard), x, len(new_covData[0]))
#     mvf_unknown = self.MVF(self.mle_array, omega1, hazard, x-1, new_covData)
#     mvf_known = self.MVF(self.mle_array, omega2, hazard, x-2, new_covData)

#     print(mvf_unknown - mvf_known - intensity)
#     return mvf_unknown - mvf_known - intensity
