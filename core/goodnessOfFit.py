import numpy as np


def PSSE(fitted, actual, intervals):
    # print(fitted[(self.n - 1) + 1:])
    # sub = np.subtract(fitted[(self.n - 1) + 1:], actual[(self.n - 1) + 1:])
    # error = np.sum(np.power(sub, 2))
    # return error

    sub = np.subtract(fitted[intervals:], actual[intervals:])
    error = np.sum(np.power(sub, 2))
    return error


class Comparison():
    """
    For comparison of model goodness-of-fit measures
    """

    def __init__(self):
        # self.meanOutUniform = None
        self.meanOut = None
        self.medianOut = None
        # self.bestMeanUniform = None
        self.bestMean = None
        self.bestMedian = None

        # self.meanOutUniformDict = {}
        # self.meanOutDict = {}

    def criticMethod(self, results, sideMenu):
        # numResults = len(results)
        llf = []
        aic = []
        bic = []
        sse = []

        self._weightSum = self.calcWeightSum(sideMenu)

        converged = 0   # number of converged models
        for key, model in results.items():
            if model.converged:
                llf.append(model.llfVal)
                aic.append(model.aicVal)
                bic.append(model.bicVal)
                sse.append(model.sseVal)
                converged += 1

        llfOut = np.zeros(converged)    # create np arrays, num of elements = num of converged
        aicOut = np.zeros(converged)
        bicOut = np.zeros(converged)
        sseOut = np.zeros(converged)

        for i in range(converged):
            llfOut[i] = self.ahp(llf, i, sideMenu.llfSpinBox, False)
            aicOut[i] = self.ahp(aic, i, sideMenu.aicSpinBox, False)
            bicOut[i] = self.ahp(bic, i, sideMenu.bicSpinBox, False)
            sseOut[i] = self.ahp(sse, i, sideMenu.sseSpinBox, False)

        ahpArray = np.array([llfOut, aicOut, bicOut, sseOut])   # array of goodness of fit arrays

        # raw values: not on a scale strictly from 0.0 to 1.0
        rawMean = np.mean(ahpArray, axis=0)     # mean of each goodness of fit measure,
                                                # for each model/metric combination
        rawMedian = np.median(ahpArray, axis=0)
        
        # Exception raised if trying to take max of empty array
        try:
            # divide by max value to normalize on scale from 0.0 to 1.0
            maxMean = np.max(rawMean)
            maxMeadian = np.max(rawMedian)
            self.meanOut = np.divide(rawMean, maxMean)
            self.medianOut = np.divide(rawMedian, maxMeadian)
        except:
            pass
        # self.meanOut = ahp_array_sum

        self.bestCombinations()

    def criticMethod_model(self, data, sideMenu):
        """
        When Pandas dataframe is passed
        """
        numRows = len(data)

        # get numpy arrays of Pandas series
        llf = data['Log-Likelihood'].values
        aic = data['AIC'].values
        bic = data['BIC'].values
        sse = data['SSE'].values

        self._weightSum = self.calcWeightSum(sideMenu)

        llfOut = np.zeros(numRows)    # create np arrays, num of elements = num of converged
        aicOut = np.zeros(numRows)
        bicOut = np.zeros(numRows)
        sseOut = np.zeros(numRows)

        for i in range(numRows):
            llfOut[i] = self.ahp(llf, i, sideMenu.llfSpinBox, False)
            aicOut[i] = self.ahp(aic, i, sideMenu.aicSpinBox, False)
            bicOut[i] = self.ahp(bic, i, sideMenu.bicSpinBox, False)
            sseOut[i] = self.ahp(sse, i, sideMenu.sseSpinBox, False)

        ahpArray = np.array([llfOut, aicOut, bicOut, sseOut])   # array of goodness of fit arrays

        # raw values: not on a scale strictly from 0.0 to 1.0
        rawMean = np.mean(ahpArray, axis=0)     # mean of each goodness of fit measure,
                                                # for each model/metric combination
        rawMedian = np.median(ahpArray, axis=0)
        
        # Exception raised if trying to take max of empty array
        try:
            # divide by max value to normalize on scale from 0.0 to 1.0
            maxMean = np.max(rawMean)
            maxMeadian = np.max(rawMedian)
            self.meanOut = np.divide(rawMean, maxMean)
            self.medianOut = np.divide(rawMedian, maxMeadian)
        except:
            pass
        # self.meanOut = ahp_array_sum

        self.bestCombinations()

    def calcWeightSum(self, sideMenu):
        return sideMenu.llfSpinBox.value() + sideMenu.aicSpinBox.value() + sideMenu.bicSpinBox.value() + sideMenu.sseSpinBox.value()

    def ahp(self, measureArray, i, spinBox, uniform):
        if uniform:
            weight = 1.0/4.0
        else:
            try:
                weight = spinBox.value()/self._weightSum
            except ZeroDivisionError:
                # all spin boxes set to zero, give equal weighting
                weight = 1.0/4.0


        if len(measureArray) > 1:
            ahp_val = (abs(measureArray[i]) - max(np.absolute(measureArray))) / (min(np.absolute(measureArray)) - max(np.absolute(measureArray))) * weight
        else:
            ahp_val = 1.0/4.0

        return ahp_val


    def bestCombinations(self):
        # store the index of model combinations that have the highest value, will bold these cells
        try:
            # self.bestMeanUniform = np.argmax(self.meanOutUniform)
            self.bestMean = np.argmax(self.meanOut)
            self.bestMedian = np.argmax(self.medianOut)
        except ValueError:
            # self.bestMeanUniform = None
            self.bestMean = None
            self.bestMedian = None
