# Covariate Software Failure and Reliability Assessment Tool (C-SFRAT)
The Covariate Software Failure and Reliability Assessment Tool (C-SFRAT) is an open source application that applies covariate software reliability models to help guide model selection and test activity allocation. The primary functions of the C-SFRAT include:
1.	Displaying model fit and failure intensity plots of selected hazard function and covariate combinations
2.	Prediction of future failures and failure intensity based on a specified testing activity profile
3.	Comparison of fitted models based on information theoretic and predictive goodness-of-fit measures with user-defined weighting
4.	Recommendations for test activity allocation to maximize defect discovery within a specified budget or minimize the total testing resources required to discover a specified number of defects.

## Installation
C-SFRAT is compatible with Windows, macOS, and Linux running Python 3.x.

Python can be installed from: https://www.python.org/downloads/.

### Libraries
The Python libraries required to run C-SFRAT are:
* Matplotlib
* NumPy
* openpyxl
* pandas
* PyQt5
* SciPy
* SymEngine
* SymPy

To install compatible versions of the libraries, first open a terminal/command prompt window and navigate to the C-SFRAT root directory. Then, run the command `pip install -r requirements.txt`.

## Running
To run C-SFRAT, first navigate to the C-SFRAT root directory in a terminal/command prompt window. Run C-SFRAT using the command `python main.py`. Depending on your Python installation, you may instead need to enter `python3 main.py`.

### Command Line Arguments
Usage: `python main.py [-v] [-d]`
* -v, --verbose Run application in verbose mode, offering more detailed information printed to the terminal
* -d, --debug Run application in debug mode, offering additional details that can help with debugging but are not required for typical use

## Folder Structure
The main folders of C-SFRAT are:
* core - contains mathematical functions used for model fitting and prediction
* datasets - contains example data sets
* models - contains hazard function definitions
* ui - contains graphical user interface layout definitions

## References
The covariate model that C-SFRAT applies was presented in:

V. Nagaraju, C. Jayasinghe, and L. Fiondella, “Optimal test activity allocation for covariate software reliability and security models,” *Journal of Systems and Software*, p. 110643, 2020.

## Acknowledgement
This material is based upon work supported by the National Science Foundation under Grant Number (#1749635).
