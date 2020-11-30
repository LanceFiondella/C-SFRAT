import pytest
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from models.discreteWeibull2 import DiscreteWeibull2
from core.dataClass import Data
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
mylogger = logging.getLogger()

mylogger.info('\n###############\nStarting ISS Model Testing\n###############')

def setup_iss(Systemdata):
    fname = "result.xlsx"
    dataResults = pd.read_excel(fname, sheet_name='Dweibull2')


    #aMLE=round(aMLE,1)
    #bMLE=round(bMLE,1)
    #cMLE=round(cMLE,1)

    ISS_list = []

    for sheet in Systemdata.sheetNames:
        rawData = Systemdata.dataSet[sheet]
        try:
            iss = ISS(data=rawData, rootAlgoName='bisect')
            iss.findParams(0)
        except:
            pass
        ISS_list.append(iss)
    return [ISS_list, aMLE, bMLE, cMLE]

fname = "model_data.xlsx"
Systemdata = Data()
Systemdata.importFile(fname)
DATA = setup_iss(Systemdata)
Results_aMLE = []
Results_bMLE = []
Results_cMLE = []
for i in range(0, len(DATA[0])):
    try:
        Results_aMLE.append((DATA[0][i].aMLE, DATA[1][i],Systemdata.sheetNames[i]))
        Results_bMLE.append((DATA[0][i].bMLE, DATA[2][i]),Systemdata.sheetNames[i])
        Results_cMLE.append((DATA[0][i].cMLE, DATA[3][i]),Systemdata.sheetNames[i])
    except:
        mylogger.info('Error in Sheet number ' + Systemdata.sheetNames[i])


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