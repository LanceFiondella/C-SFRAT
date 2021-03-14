import pytest
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from models.discreteWeibull2 import DiscreteWeibull2
from core.dataClass import Data
import configparser
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
mylogger = logging.getLogger()

mylogger.info('\n###############\nStarting  Model Testing\n###############')

def DWSetup(Systemdata):
    MetricCombos = [[],['E'],['F'],['C'],['E','F'],['F','C'],['E','C'],['E','F','C']]
    DWresults = {}
    config = configparser.ConfigParser()
    config.read('config.ini')
    for metricname in MetricCombos:
        DW = DiscreteWeibull2(data=Systemdata.getFullData(),metricNames=metricname,config=config)
        DW.runEstimation(DW.covariateData)
        if (metricname == []):
            metricname = ['-']
        DWresults[listToString(metricname)] = {"b":DW.modelParameters[0],"Beta":DW.betas , "Omega": DW.omega,"LL" : DW.llfVal}
    return DWresults


def listToString(s):
    # initialize an empty string
    str1 = ""
    
    # traverse in the string
    for ele in s:
        str1 += ele
        
        # return string
    return str1


fname = "ds1.csv"
SystemdataDS1 = Data()
SystemdataDS1.importFile(fname)
DWresultsDS1 = DWSetup(SystemdataDS1)

fname = "ds2.csv"
SystemdataDS2 = Data()
SystemdataDS2.importFile(fname)
DWresultsDS2 = DWSetup(SystemdataDS2)




@pytest.mark.parametrize("test_input,expected,SheetName", Results_aMLE)
def test_iss_a_mle(test_input, expected,SheetName):
    assert abs(test_input - expected) < 10**-3


@pytest.mark.parametrize("test_input,expected,SheetName", Results_bMLE)
def test_iss_b_mle(test_input, expected,SheetName):
    assert abs(test_input - expected) < 10**-3

@pytest.mark.parametrize("test_input,expected,SheetName", Results_cMLE)
def test_iss_c_mle(test_input, expected,SheetName):
    assert abs(test_input - expected) < 10**-3

def test_name():
    for iss in DATA[0]:
        try:
            assert iss.name == "ISS"
        except:
            pass