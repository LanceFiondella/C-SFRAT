def setDataView(self, viewType, index):
    """Sets the data to be displayed.

    Called whenever a menu item is changed
    Two options for viewType: "view" or "sheet". The index
    controls which option of the selected viewType is selected.

    Args:
        viewType: String that determines if plot type, trend test, or sheet
            is set.
        index: Index (int) that determines which plot type, trend test, or
            sheet to display. Dependent on viewType.
    """

    if self.data.getData() is not None:
        if viewType == "view":
            self.setRawDataView(index)
            self.dataViewIndex = index
        elif viewType == "sheet":
            self.changeSheet(index)

def changeSheet(self, index):
    """Changes the current sheet displayed.
    Handles data that needs to be changed when sheet changes.

    Args:
        index: The index of the sheet (int).
    """
    self.data.currentSheet = index      # store
    self.data.max_interval = self.data.n
    self._main.tab1.sideMenu.updateSlider(self.data.n)
    # create plot
    self.updateUI()
    self.redrawPlot(1)
    self.setMetricList()

def updateUI(self):
    """Updates plots, tables, side menus.

    Should be called explicitly.
    """
    self.setDataView("view", self.dataViewIndex)

def setRawDataView(self, index):
    """Creates MVF or intensity plot, based on index.

    Args:
        index: Integer that controls which plot to create. 0 creates MVF
            plot, 1 creates intensity plot.
    """
    self._main.tab1.plotAndTable.tableWidget.setModel(self.data.getDataModel())
    dataframe = self.data.getData()
    # self.plotSettings.plotType = "step"

    if self.dataViewIndex == 0:
        # MVF
        self.mvf.setChecked(True)
        self.createMVFPlot(dataframe)

        # disable reliability spin box, enable failure spin box
        self._main.tab2.sideMenu.reliabilitySpinBox.setDisabled(True)
        self._main.tab2.sideMenu.failureSpinBox.setEnabled(True)

    if self.dataViewIndex == 1:
        # Intensity
        self.intensity.setChecked(True)
        self.createIntensityPlot(dataframe)

        # disable failure spin box, enable reliability spin box
        self._main.tab2.sideMenu.failureSpinBox.setDisabled(True)
        self._main.tab2.sideMenu.reliabilitySpinBox.setEnabled(True)

    # redraw figures
    self.ax2.legend()
    self.redrawPlot(1)
    self.redrawPlot(2)

def createMVFPlot(self, dataframe):
    """Creates MVF plots for tabs 1 and 2.

    Creates step plot for imported data. Tab 2 plot only displayed if
    estimation is complete. For fitted data, creates either a step or
    smooth plot, depending on what has been specified by the user in the
    menu bar. Called by setRawDataView method.
    """
    # self.plotSettings.plotType = "plot" # if continous
    # self.plotSettings.plotType = "step" # if step

    # save previous plot type, always want observed data to be step plot
    previousPlotType = self.plotSettings.plotType

    # tab 1 plot
    self.plotSettings.plotType = "step"
    self.ax = self.plotSettings.generatePlot(self.ax, dataframe['T'], dataframe["CFC"],
                                                title="", xLabel="Intervals", yLabel="Cumulative failures")

    # tab 2 plot
    if self.estimationComplete:
        self.ax2 = self.plotSettings.generatePlot(self.ax2, dataframe['T'], dataframe["CFC"],
                                                    title="", xLabel="Intervals", yLabel="Cumulative failures")

        self.plotSettings.plotType = previousPlotType   # want model fits to be plot type specified by user

        # add vertical line at last element of original data
        self.ax2.axvline(x=dataframe['T'].iloc[-1], color='red', linestyle='dotted')


        # TEMPORARY
        # for displaying predictions in tab 2 table
        prediction_list = [0]
        model_name_list = ["Interval"]


        # self.plotSettings.plotType = "step"
        # model name and covariate combination
        for modelName in self.selectedModelNames:
            # add line for model if selected
            model = self.estimationResults[modelName]

            # check if prediction is specified
            if self._main.tab2.sideMenu.failureSpinBox.value() > 0:
                x, mvf_array = self.runPredictionMVF(model, self._main.tab2.sideMenu.failureSpinBox.value())
                self.plotSettings.addLine(self.ax2, x, mvf_array, modelName)

                # TEMPORARY
                prediction_list[0] = x
                prediction_list.append(mvf_array)
                model_name_list.append(modelName)
            else:
                self.plotSettings.addLine(self.ax2, model.t, model.mvf_array, modelName)

        # TEMPORARY
        # check if prediction is specified
            if self._main.tab2.sideMenu.failureSpinBox.value() > 0:
                self._main.tab2.updateTable_prediction(prediction_list, model_name_list, 0)
                
def createIntensityPlot(self, dataframe):
    """Creates intensity plots for tabs 1 and 2.

    Creates step plot for imported data. Tab 2 plot only displayed if
    estimation is complete. For fitted data, creates either a step or
    smooth plot, depending on what has been specified by the user in the
    menu bar. Called by setRawDataView method.
    """
    # need to change plot type to "bar" for intensity view, but want model result lines
    # to use whatever plot type had been selected
    # save the previous plot type, use it after bar plot created

    previousPlotType = self.plotSettings.plotType
    self.plotSettings.plotType = "bar"
    # self.plotSettings.plotType = "step"

    # disable trend tests when displaying imported data
    # self._main.tab1.sideMenu.testSelect.setDisabled(True)
    # self._main.tab1.sideMenu.confidenceSpinBox.setDisabled(True)

    self.ax = self.plotSettings.generatePlot(self.ax, dataframe['T'], dataframe.iloc[:, 1],
                                                title="", xLabel="Intervals", yLabel="Failures")
    if self.estimationComplete:
        self.ax2 = self.plotSettings.generatePlot(self.ax2, dataframe['T'], dataframe['FC'],
                                                    title="", xLabel="Intervals", yLabel="Failures")
        self.plotSettings.plotType = previousPlotType


        # TEMPORARY
        # for displaying predictions in tab 2 table
        prediction_list = [0]
        model_name_list = ["Interval"]


        # model name and covariate combination
        for modelName in self.selectedModelNames:
            # add line for model if selected
            model = self.estimationResults[modelName]

            # check if prediction is specified
            if self._main.tab2.sideMenu.reliabilitySpinBox.value() > 0.0:
                x, intensity_array, interval = self.runPredictionIntensity(model, self._main.tab2.sideMenu.reliabilitySpinBox.value())
                self.plotSettings.addLine(self.ax2, x, intensity_array, modelName)

                # TEMPORARY
                prediction_list[0] = x
                prediction_list.append(intensity_array)
                model_name_list.append(modelName)
            else:
                self.plotSettings.addLine(self.ax2, model.t, model.intensityList, modelName)

        # TEMPORARY
        # check if prediction is specified
            if self._main.tab2.sideMenu.reliabilitySpinBox.value() > 0.0:
                self._main.tab2.updateTable_prediction(prediction_list, model_name_list, 1)







    # def keyPressEvent(self, e):
    #     """
    #     For copying tab 3 table

    #     https://stackoverflow.com/questions/24971305/copy-pyqt-table-selection-including-column-and-row-headers
    #     """
    #     if (e.modifiers() & Qt.ControlModifier):
    #         # selected = self.tab3.table.selectedIndexes()

    #         selected1 = self.tab3.table.selectionModel()
    #         print(selected1)
    #         print(selected1.selection())

    #         selected = selected1.selection()
    #         print(selected)

    #         if e.key() == Qt.Key_C: #copy
    #             s = '\t'+"\t".join([str(self.table.horizontalHeaderItem(i).text()) for i in range(selected[0].index(), selected[-1].index()+1)])
    #             s = s + '\n'

    #             for r in range(selected[0].topRow(), selected[0].bottomRow()+1):
    #                 s += self.table.verticalHeaderItem(r).text() + '\t'
    #                 for c in range(selected[0].leftColumn(), selected[0].rightColumn()+1):
    #                     try:
    #                         s += str(self.table.item(r ,c).text()) + "\t"
    #                     except AttributeError:
    #                         s += "\t"
    #                 s = s[:-1] + "\n" #eliminate last '\t'
    #             self.clip.setText(s)
