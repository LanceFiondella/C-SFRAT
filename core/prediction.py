import logging as log

import numpy as np


def prediction_mvf(model, failures, covariate_data, effortDict):
    """
    effortDict: dictionary containing all prediction effort spin box widgets,
        indexed by covariate string
    """

    total_points = model.n + failures

    new_array = []
    for cov in model.metricNames:
        value = effortDict[cov].value()
        new_array.append(np.full(failures, value))

    if model.numCovariates == 0:
        combined_array = np.concatenate((covariate_data, np.array(new_array)))
    else:
        combined_array = np.concatenate((covariate_data, np.array(new_array)), axis=1)

    newHazard = np.array([model.hazardNumerical(i, model.modelParameters) for i in range(model.n, total_points)])  # calculate new values for hazard function
    hazard = np.concatenate((model.hazard_array, newHazard))


    ## VERIFY OMEGA VALUE, should we continue updating?

    omega = model.calcOmega(hazard, model.betas, combined_array)

    mvf_array = np.array([model.MVF(model.mle_array, omega, hazard, dataPoints, combined_array) for dataPoints in range(total_points)])
    x = np.concatenate((model.t, np.arange(model.n + 1, total_points + 1)))

    return (x, mvf_array)

def prediction_psse(model, data):
    """
    Prediction function used for PSSE. Imported covariate data is used.
    """

    total_points = data.max_interval
    full_data = data.getData()
    covariateData = np.array([full_data[name] for name in model.metricNames])
    newHazard = np.array([model.hazardNumerical(i, model.modelParameters) for i in range(model.n, total_points)])  # calculate new values for hazard function
    hazard = np.concatenate((model.hazard_array, newHazard))


    ## VERIFY OMEGA VALUE, should we continue updating?

    omega = model.calcOmega(hazard, model.betas, covariateData)
    mvf_array = np.array([model.MVF(model.mle_array, omega, hazard, dataPoints, covariateData) for dataPoints in range(total_points)])
    return mvf_array

def prediction_intensity(model, intensity, covariate_data, effortDict):
    mvf_list = model.mvf_array.tolist()

    for i in range(1, 100):
        total_points = model.n + i

        new_array = []
        j = 0
        for cov in model.metricNames:
            value = effortDict[cov].value()
            new_array.append(np.full(i, value))
            j += 1

        if model.numCovariates == 0:
            combined_array = np.concatenate((covariate_data, np.array(new_array)))
        else:
            combined_array = np.concatenate((covariate_data, np.array(new_array)), axis=1)

        newHazard = np.array([model.hazardNumerical(j, model.modelParameters) for j in range(model.n, total_points)])  # calculate new values for hazard function
        hazard = np.concatenate((model.hazard_array, newHazard))

        ## VERIFY OMEGA VALUE, should we continue updating?
        omega = model.calcOmega(hazard, model.betas, combined_array)

        #### IGNORE IF 0 !!!!!! ####

        mvf_list.append(model.MVF(model.mle_array, omega, hazard, total_points - 1, combined_array))
        calculated_intensity = mvf_list[-1] - mvf_list[-2]
        log.info("calculated intensity:", calculated_intensity)
        log.info("desired intensity:", intensity)
        if calculated_intensity < intensity:
            log.info("desired failure intensity reached in {0} intervals".format(i))
            x = np.concatenate((model.t, np.arange(model.n + 1, len(mvf_list) + 1)))
            return (x, model.intensityFit(mvf_list), i)

    log.info("desired failure intensity not reached within 100 intervals")
    return (model.t, model.intensityList, 0)
