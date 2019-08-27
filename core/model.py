from abc import ABC, abstractmethod, abstractproperty


class Model(ABC):
    def __init__(self, *args, **kwargs):
        '''
        Initialize Model

        Keyword Args:
            data: Pandas dataframe with all required columns
        '''
        self.data = kwargs["data"]

    ##############################################
    #Properties/Members all models must implement#
    ##############################################
    @abstractproperty
    def name(self):
        '''
        Name of model (string)
        '''
        return "Generic Model"

    @abstractproperty
    def converged(self):
        '''
        Indicates whether model has converged (Bool)
        Must be set after parameters are calculated
        '''
        return True

    #################################################
    #Methods that must be implemented by all models#
    ################################################
    @abstractmethod
    def initialEstimates(self):
        pass

    @abstractmethod
    def LLF_sym(self):
        pass

    @abstractmethod
    def convertSym(self):
        pass

    @abstractmethod
    def optimizeSolution(self):
        pass

    @abstractmethod
    def calcHazard(self):
        pass

    @abstractmethod
    def calcOmega(self):
        pass