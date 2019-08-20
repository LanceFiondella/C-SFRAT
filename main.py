#!/usr/bin/env python3

import argparse  # to handle command line arguments
from PyQt5.QtWidgets import QApplication  # for UI
from PyQt5.QtCore import Qt  # for UI
import logging as log  # to handle debug output
import sys

from ui.mainWindow import MainWindow


def main(debug):
    log.info("Hello From Main")
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

    if args.verbose:
        # if using verbose output set logging to display level and message
        # print anything level debug or higher
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        log.info("Using Verbose output.")
    if args.debug:
        debug = True
    else:
        # if not using verbose output only print errors and warnings
        log.basicConfig(format="%(levelname)s: %(message)s")

    main(debug)
