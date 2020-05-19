#!/usr/bin/env python3

import argparse  # to handle command line arguments
import logging as log # to handle debug output
import sys

from PyQt5.QtWidgets import QApplication  # for UI
from PyQt5.QtCore import Qt  # for UI

from ui.mainWindow import MainWindow

# log = log.getLogger("ui.mainWindow")

def main(debug):
    """
    Description to be created at a later time.

    Args
        debug : 
    """
    log.info("Starting Covariate Tool application.")
    app = QApplication(sys.argv)
    mainWindow = MainWindow(debug)
    sys.exit(app.exec_())


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description='Covariate Tool')
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
