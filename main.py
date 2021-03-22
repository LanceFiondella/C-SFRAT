#!/usr/bin/python

"""File that is run to begin the Covariate Tool application.

This file is run using the command line, with optional arguments.
These arguments allow the application to be run in verbose mode,
which provides additional information on calculations, or in debug mode,
which prints informations about application state and events that
a typical user would not need to worry about.

    Typical usage example (on command line):

    python main.py
    python main.py -v
    python main.py -d

    The first line will run the application in the default mode,
    with no additional information printed to the command line.
    The second line will run the application in verbose mode.
    The third line will run the application in debug mode.
"""

import argparse  # to handle command line arguments
import logging as log  # to handle debug output
import sys

from PyQt5.QtWidgets import QApplication  # for UI

from ui.mainWindow import MainWindow


def main(debug):
    """
    Begins running application and creates instance of MainWindow class.

    Args
        debug: Boolean indicating if debug mode is active or not
    """
    log.info("Starting C-SFRAT.")
    app = QApplication(sys.argv)
    mainWindow = MainWindow(debug)
    sys.exit(app.exec_())


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description='C-SFRAT')
    parser.add_argument("-v", '--verbose', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()
    debug = False

    if args.debug:
        debug = True
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        log.debug("In DEBUG mode.")
    elif args.verbose:

        # if using verbose output set logging to display level and message
        # print anything level debug or higher

        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
        log.info("Using verbose output.")
    else:
        # if not using verbose output only print errors and warnings
        log.basicConfig(format="%(levelname)s: %(message)s")

    main(debug)
