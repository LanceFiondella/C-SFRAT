from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
import numpy as np

class TrendTest(ABC):
    name = 'Trend test name not set'
    xAxisLabel = 'X'
    yAxisLabel = 'Y'

    def __init__(self):
        """
        All Trend Tests should be subclasses of this class
        """
        pass

    @abstractmethod
    def run(self, data):
        """
        Run method must be implemented
        Args:
            data: Pandas Dataframe of raw data
        Returns:
            pandas DataFrame with only 2 columns for x and y axes respectively
        """
        return pd.DataFrame({'X-axis': [0], 'Y-axis': [0]})

    # def convertDataframe(self, data):
    #     pass

class LaplaceTest(TrendTest):
    """
    Laplace Trend Test
    """
    name = 'Laplace Trend Test'
    xAxisLabel = 'Failure Number'
    yAxisLabel = 'Laplace Test Statistic'

    def __init__(self):
        super().__init__()

    def run(self, data):
        laplace = pd.Series(0)  # 0 gives workable series
        cum_sum = np.cumsum(data['FT'])
        for i in range(1, len(data)):
            # cur_sum = sum(data['FT'][:i])
            cur_sum = cum_sum[i-1]
            laplace[i] = ((((1/(i))*cur_sum) - (data['FT'][i]/2)) /
                          (data['T'][i]*(1/(12*(i))**(0.5))))

        return pd.DataFrame({'X': data['FN'], 'Y': laplace})


# class AverageTest(TrendTest):
#     """
#     Running Arithmetic Average
#     """
#     name = 'Running Arithmetic Average'
#     xAxisLabel = 'Failure Number'
#     yAxisLabel = 'Running Average of Interfailure Times'

#     def __init__(self):
#         super().__init__()

#     def run(self, data):
#         avg = pd.Series(0)
#         for i in range(len(data)):
#             avg[i] = sum(data['IF'][0:i+1])/(i+1)

#         return pd.DataFrame({'X': data['FN'], 'Y': avg})
