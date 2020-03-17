#!/usr/bin/env python3

import argparse  # to handle command line arguments
import logging # to handle debug output
import sys

from PyQt5.QtWidgets import QApplication  # for UI
from PyQt5.QtCore import Qt  # for UI

from ui.mainWindow import MainWindow

# log = logging.getLogger("ui.mainWindow")

def main(debug):
    """
    Description to be created at a later time.

    Args
        debug : 
    """
    logging.info("Starting Covariate Tool application.")
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
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)
        logging.debug("In DEBUG mode.")
    elif args.verbose:
        # if using verbose output set logging to display level and message
        # print anything level debug or higher
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
        logging.info("Using verbose output.")
    else:
        # if not using verbose output only print errors and warnings
        logging.basicConfig(format="%(levelname)s: %(message)s")

    main(debug)
