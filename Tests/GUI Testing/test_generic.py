import pytest
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from core.dataClass import Data
from models import discreteWeibull2,geometric,negativeBinomial2
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
mylogger = logging.getLogger()
mylogger.info('\n###############\nStarting Generic Testing\n###############')


"""
ISSUES: models.modelList is displaying the correct contents for some reason. It works when tested in the console but not on this script
HARDCODED right now so it needs to be changed
"""

ModelDict = {'discreteWeibull2': discreteWeibull2.DiscreteWeibull2, 'geometric' : geometric.Geometric, 'negativeBinomial2': negativeBinomial2.NegativeBinomial2}

#DATA SETUP
def setup_data():
    return Data()

sample_data = setup_data()
sample_data.importFile("ds1.csv")


#Model List Setup
print(ModelDict.values())
sample_models = []
for values in ModelDict.values():
    sample_models.append(values(data=pd.read_csv("ds1.csv"), rootAlgoName='bisect'))

print(sample_models)


#Testing data class property types
def test_data_sheets():
    assert type(sample_data.sheetNames) is list
    for sheet in sample_data.sheetNames:
        assert type(sheet) is str


def test_data_current_sheet():
    assert type(sample_data._currentSheet) is int


def test_data_property_current_sheet():
    assert sample_data._currentSheet == sample_data.currentSheet


def test_getData():
    for sheetnum in range(0, len(sample_data.sheetNames)):
        sample_data.currentSheet = sheetnum
        assert sample_data.getData() is dict

"""
def test_data_currentSheet_setter():
    sample_data.currentSheet(10)
    assert sample_data.currentSheet == 10
    sample_data.currentSheet(100)
    assert sample_data.currentSheet == 0
    sample_data.currentSheet(len(sample_data.sheetNames))
    assert sample_data.currentSheet == 0
    sample_data.currentSheet(-1)
    assert sample_data.currentSheet == 0


def test_data_dataSet():
    assert type(sample_data.dataSet) is dict
    for key, value in sample_data.dataSet:
        assert type(key) is str
"""


#Testing model property types
def test_models_names():
    assert type(sample_models[0].name) is str


def test_models_params():
    for model in sample_models:
        assert type(model.params) is dict
        for key, value in model.params:
            assert type(key) is str
            assert type(value) is float


def test_models_root_name():
    for model in sample_models:
        assert type(model.rootAlgoName) is str


def test_models_coverged():
    for model in sample_models:
        assert type(model.converged) is bool


#Testing functions in models
def test_models_findParams():
    for model in sample_models:
        assert 'findParams' in dir(model)


def test_models_predict():
    for model in sample_models:
        assert 'predict' in dir(model)


def test_models_reliability():
    for model in sample_models:
        assert 'reliability' in dir(model)


def test_models_lnL():
    for model in sample_models:
        assert 'lnL' in dir(model)


def test_models_MVF():
    for model in sample_models:
        assert 'MVF' in dir(model)


def test_models_MVFPlot():
    for model in sample_models:
        assert 'MVFPlot' in dir(model)


def test_models_MTTFPlot():
    for model in sample_models:
        assert 'MTTFPlot' in dir(model)


def test_models_FIPlot():
    for model in sample_models:
        assert 'FIPlot' in dir(model)


def test_models_relGrowthPlot():
    for model in sample_models:
        assert 'relGrowthPlot' in dir(model)