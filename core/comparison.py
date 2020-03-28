import numpy as np

class Comparison():
    """
    For comparison of model goodness-of-fit measures
    """

    def __init__(self):
        self.meanOutUniform = None
        self.meanOut = None
        self.bestMeanUniform = None
        self.bestMean = None

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
            llfOutUniform[i] = self.ahpNegative(llf, i, sideMenu.llfSpinBox, True)
            aicOutUniform[i] = self.ahp(aic, i, sideMenu.aicSpinBox, True)
            bicOutUniform[i] = self.ahp(bic, i, sideMenu.bicSpinBox, True)
            sseOutUniform[i] = self.ahp(sse, i, sideMenu.sseSpinBox, True)
            llfOut[i] = self.ahpNegative(llf, i, sideMenu.llfSpinBox, False)
            aicOut[i] = self.ahp(aic, i, sideMenu.aicSpinBox, False)
            bicOut[i] = self.ahp(bic, i, sideMenu.bicSpinBox, False)
            sseOut[i] = self.ahp(sse, i, sideMenu.sseSpinBox, False)

        ahpArrayUniform = np.array([llfOutUniform, aicOutUniform, bicOutUniform, sseOutUniform])
        ahpArray = np.array([llfOut, aicOut, bicOut, sseOut])   # array of goodness of fit arrays

        self.meanOutUniform = np.mean(ahpArrayUniform, axis=0)
        self.meanOut = np.mean(ahpArray, axis=0)  # mean of each goodness of fit measure,
                                                    # for each model/metric combination
        self.bestCombinations()

    def bestCombinations(self):
        # store the index of model combinations that have the highest value, will bold these cells
        self.bestMeanUniform = np.argmax(self.meanOutUniform)
        self.bestMean = np.argmax(self.meanOut)

    def ahpNegative(self, measureList, i, spinBox, uniform):
        if uniform:
            weight = 1.0/4.0
        else:
            try:
                weight = spinBox.value()/self._weightSum
            except ZeroDivisionError:
                weight = 1.0/4.0
        
        ahp_val = (measureList[i] - min(measureList)) / (max(measureList) - min(measureList)) * weight

        return ahp_val

    def ahp(self, measureList, i, spinBox, uniform):
        """
        negative is bool. Calculating weight for LLF is different because its values are negative. Specified
        by True, otherwise False.

        uniform is bool. If calculating with uniform (no) weighting, uniform = True.
        """
        # print("num =", measureList[i] - min(measureList))
        # print("den =", max(measureList) - min(measureList))
        # print("weight =", spinBox.value()/self.weightSum)

        if uniform:
            weight = 1.0/4.0
        else:
            try:
                weight = spinBox.value()/self._weightSum
            except ZeroDivisionError:
                weight = 1.0/4.0

        ahp_val = (measureList[i] - max(measureList)) / (min(measureList) - max(measureList)) * weight
        
        return ahp_val

    def calcWeightSum(self, sideMenu):
        return sideMenu.llfSpinBox.value() + sideMenu.aicSpinBox.value() + sideMenu.bicSpinBox.value() + sideMenu.sseSpinBox.value()