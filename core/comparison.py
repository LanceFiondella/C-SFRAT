import numpy as np


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

    def goodnessOfFit(self, results, sideMenu):
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

        llfOutUniform = np.zeros(converged)    # create np arrays, num of elements = num of converged
        aicOutUniform = np.zeros(converged)
        bicOutUniform = np.zeros(converged)
        sseOutUniform = np.zeros(converged)

        llfOut = np.zeros(converged)    # create np arrays, num of elements = num of converged
        aicOut = np.zeros(converged)
        bicOut = np.zeros(converged)
        sseOut = np.zeros(converged)

        for i in range(converged):
            # llfOutUniform[i] = self.ahpNegative(llf, i, sideMenu.llfSpinBox, True)
            # llfOutUniform[i] = self.ahp(llf, i, sideMenu.llfSpinBox, True)
            # aicOutUniform[i] = self.ahp(aic, i, sideMenu.aicSpinBox, True)
            # bicOutUniform[i] = self.ahp(bic, i, sideMenu.bicSpinBox, True)
            # sseOutUniform[i] = self.ahp(sse, i, sideMenu.sseSpinBox, True)
            # llfOut[i] = self.ahpNegative(llf, i, sideMenu.llfSpinBox, False)
            llfOut[i] = self.ahp(llf, i, sideMenu.llfSpinBox, False)
            aicOut[i] = self.ahp(aic, i, sideMenu.aicSpinBox, False)
            bicOut[i] = self.ahp(bic, i, sideMenu.bicSpinBox, False)
            sseOut[i] = self.ahp(sse, i, sideMenu.sseSpinBox, False)

        # ahpArrayUniform = np.array([llfOutUniform, aicOutUniform, bicOutUniform, sseOutUniform])
        ahpArray = np.array([llfOut, aicOut, bicOut, sseOut])   # array of goodness of fit arrays

        # print(ahpArray)
        # print(np.sum(ahpArray, axis=0))

        ahp_array_sum = np.sum(ahpArray, axis=0)

        # self.meanOutUniform = np.mean(ahpArrayUniform, axis=0)
        self.meanOut = np.mean(ahpArray, axis=0)  # mean of each goodness of fit measure,
                                                    # for each model/metric combination
        self.medianOut = np.median(ahpArray, axis=0)

        self.meanOut = ahp_array_sum
        # print(self.meanOut)
        # print(self.medianOut)

        # store results in dictionary indexed by combination name
        # for key, model in results.items():
        #     count = 0
        #     self.meanOutUniformDict[key] = self.meanOutUniform[count]
        #     self.meanOutDict[key] = self.meanOut[count]
        #     count += 1

        # print(self.meanOutUniformDict)

        self.bestCombinations()

    def calcWeightSum(self, sideMenu):
        return sideMenu.llfSpinBox.value() + sideMenu.aicSpinBox.value() + sideMenu.bicSpinBox.value() + sideMenu.sseSpinBox.value()

    # def ahpNegative(self, measureList, i, spinBox, uniform):
    #     """
    #     Calculating weight for LLF is different because its values are negative.
    #     """
    #     if uniform:
    #         weight = 1.0/4.0
    #     else:
    #         try:
    #             weight = spinBox.value()/self._weightSum
    #         except ZeroDivisionError:
    #             weight = 1.0/4.0

    #     if len(measureList) > 1:
    #         # ahp_val = (measureList[i] - max(measureList)) / (min(measureList) - max(measureList)) * weight
    #         ahp_val = (measureList[i] - min(measureList)) / (max(measureList) - min(measureList)) * weight
    #     else:
    #         ahp_val = 1.0

    #     return ahp_val

    def ahp(self, measureList, i, spinBox, uniform):
        if uniform:
            weight = 1.0/4.0
        else:
            try:
                weight = spinBox.value()/self._weightSum
            except ZeroDivisionError:
                # all spin boxes set to zero, give equal weighting
                weight = 1.0/4.0

        # try:
        #     ahp_val = (measureList[i] - max(measureList)) / (min(measureList) - max(measureList)) * weight
        # except ZeroDivisionError:
        #     # no difference between min and max of measure list
        #     # could mean that estimation was only run on one model
        #     ahp_val = .250

        if len(measureList) > 1:
            ahp_val = (abs(measureList[i]) - max(np.absolute(measureList))) / (min(np.absolute(measureList)) - max(np.absolute(measureList))) * weight
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
