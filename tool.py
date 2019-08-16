# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tool.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

# matplotlib for plots
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# estimation setting window
import estimation_setting

# math that does covariate calculations
import covariate

# global variables
import global_variables as gv

# for importing csv failure data
import csv, codecs, threading
import os

#----------------------------------------------------------------#
#                       TO DO:
# simplify ui code, make into separate methods/classes
# how will table data be entered?
#   ignore first line? only parse floats?
# what file should be the "main" file? should the main ui file
#   call the functions?
# make sure everything that needs to be is a np array, not list
# select hazard function
# MSE vs SSE?
# make some of the covariate variables global? reduce the number
#   of parameters needed to pass to methods
#----------------------------------------------------------------#

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 720)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        
        self.init_tabs()
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.init_menubar()

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.l = DataLoaded()
        self.l.loaded.connect(self.data_loaded_slot)     # once data is loaded, 

    def init_tabs(self):
        MainWindow.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setUsesScrollButtons(False)
        self.tabWidget.setObjectName("tabWidget")
        
        self.init_tab1()
        self.init_tab2()
        # self.init_tab3()
        # self.init_tab4()
        self.init_tab5()
        self.init_tab6()

        self.gridLayout_2.addWidget(self.tabWidget, 0, 0, 1, 1)

    def init_tab1(self):
        # faults tab
        self.tab1_fault = QtWidgets.QWidget()
        self.tab1_fault.setObjectName("tab1_fault")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.tab1_fault)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.table_layout = QtWidgets.QVBoxLayout()

        # table
        self.table_layout.setObjectName("table_layout")
        self.table_area_label = QtWidgets.QLabel(self.tab1_fault)
        self.table_area_label.setObjectName("table_area_label")
        self.table_layout.addWidget(self.table_area_label)
        # self.tableWidget = QtWidgets.QTableWidget(self.tab1_fault)

        self.model =  QtGui.QStandardItemModel()
        # self.tableWidget = DataTable()
        self.tableView = QtWidgets.QTableView()
        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableView.sizePolicy().hasHeightForWidth())
        self.tableView.setSizePolicy(sizePolicy)
        self.tableView.setModel(self.model)
        self.tableView.setObjectName("tableView")
        # self.tableWidget.setColumnCount(0)
        # self.tableWidget.setRowCount(0)
        self.table_layout.addWidget(self.tableView)
        self.horizontalLayout.addLayout(self.table_layout)
        self.tableView.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)  # set read only

        self.graph_file = QtWidgets.QVBoxLayout()
        self.graph_file.setObjectName("graph_file")
        self.fault_graph_label = QtWidgets.QLabel(self.tab1_fault)
        self.fault_graph_label.setObjectName("fault_graph_label")
        self.graph_file.addWidget(self.fault_graph_label)

        # MVF graph
        # self.fault_graph = QtWidgets.QGraphicsView(self.tab1_fault)
        self.fault_graph = PlotCanvas(width=5, height=4)
        # self.fault_graph.plot()

        # intensity graph
        self.fault_intensity_graph = PlotCanvas(width=5, height=4)

        # graph tabs
        self.fault_graph_tabs = QtWidgets.QTabWidget()
        self.fault_graph_tabs.setAutoFillBackground(False)
        self.fault_graph_tabs.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.fault_graph_tabs.setUsesScrollButtons(False)
        self.fault_graph_tabs.setObjectName("fault_graph_tabs")

        self.fault_graph_mvf_tab = QtWidgets.QWidget()
        self.fault_graph_mvf_tab.setObjectName("fault_graph_mvf_tab")

        self.fault_graph_intensity_tab = QtWidgets.QWidget()
        self.fault_graph_intensity_tab.setObjectName("fault_graph_intensity_tab")

        self.fault_graph_tabs.addTab(self.fault_graph_mvf_tab, "")
        self.fault_graph_tabs.addTab(self.fault_graph_intensity_tab, "")

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)    # want graph and table to both stretch
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fault_graph_tabs.sizePolicy().hasHeightForWidth())
        self.fault_graph_tabs.setSizePolicy(sizePolicy)

        self.graph_file.addWidget(self.fault_graph_tabs)

        self.fault_mvf_layout = QtWidgets.QHBoxLayout(self.fault_graph_mvf_tab)
        self.fault_intensity_layout = QtWidgets.QHBoxLayout(self.fault_graph_intensity_tab)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fault_graph.sizePolicy().hasHeightForWidth())
        self.fault_graph.setSizePolicy(sizePolicy)
        self.fault_graph.setObjectName("fault_graph")
        self.fault_mvf_layout.addWidget(self.fault_graph)
        self.fault_intensity_layout.addWidget(self.fault_intensity_graph)


        self.selected_file_label = QtWidgets.QLabel(self.tab1_fault)
        self.selected_file_label.setObjectName("selected_file_label")
        self.graph_file.addWidget(self.selected_file_label)
        self.file_name = QtWidgets.QLineEdit(self.tab1_fault)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.file_name.sizePolicy().hasHeightForWidth())
        self.file_name.setSizePolicy(sizePolicy)
        self.file_name.setObjectName("file_name")
        self.file_name.setReadOnly(True)
        self.graph_file.addWidget(self.file_name)
        self.horizontalLayout.addLayout(self.graph_file)
        self.tabWidget.addTab(self.tab1_fault, "")

    def init_tab2(self):
        # estimation tab
        self.tab2_est = QtWidgets.QWidget()
        self.tab2_est.setObjectName("tab2_est")
        self.tab2_layout = QtWidgets.QGridLayout(self.tab2_est)
        self.tab2_layout.setObjectName("tab2_layout")
        self.tab2_graph_label = QtWidgets.QLabel(self.tab2_est)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab2_graph_label.sizePolicy().hasHeightForWidth())
        self.tab2_graph_label.setSizePolicy(sizePolicy)
        self.tab2_graph_label.setObjectName("tab2_graph_label")
        self.tab2_layout.addWidget(self.tab2_graph_label, 0, 0, 1, 1)
        self.tab2_vertical = QtWidgets.QVBoxLayout()
        self.tab2_vertical.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.tab2_vertical.setSpacing(5)
        self.tab2_vertical.setObjectName("tab2_vertical")
        self.tab2_gof_label = QtWidgets.QLabel(self.tab2_est)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab2_gof_label.sizePolicy().hasHeightForWidth())
        self.tab2_gof_label.setSizePolicy(sizePolicy)
        self.tab2_gof_label.setObjectName("tab2_gof_label")
        self.tab2_vertical.addWidget(self.tab2_gof_label)
        self.tab2_form = QtWidgets.QFormLayout()
        self.tab2_form.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.tab2_form.setContentsMargins(-1, -1, -1, 20)
        self.tab2_form.setHorizontalSpacing(15)
        self.tab2_form.setObjectName("tab2_form")
        self.tab2_llf_label = QtWidgets.QLabel(self.tab2_est)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab2_llf_label.sizePolicy().hasHeightForWidth())
        self.tab2_llf_label.setSizePolicy(sizePolicy)
        self.tab2_llf_label.setObjectName("tab2_llf_label")
        self.tab2_form.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.tab2_llf_label)

        # LLF line edit
        self.tab2_llf_line_edit = QtWidgets.QLineEdit(self.tab2_est)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab2_llf_line_edit.sizePolicy().hasHeightForWidth())
        self.tab2_llf_line_edit.setSizePolicy(sizePolicy)
        self.tab2_llf_line_edit.setObjectName("tab2_llf_line_edit")
        self.tab2_llf_line_edit.setReadOnly(True)
        self.tab2_form.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.tab2_llf_line_edit)

        # AIC line edit
        self.tab2_aic_line_edit = QtWidgets.QLineEdit(self.tab2_est)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab2_aic_line_edit.sizePolicy().hasHeightForWidth())
        self.tab2_aic_line_edit.setSizePolicy(sizePolicy)
        self.tab2_aic_line_edit.setObjectName("tab2_aic_line_edit")
        self.tab2_aic_line_edit.setReadOnly(True)
        self.tab2_form.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.tab2_aic_line_edit)

        # BIC line edit
        self.tab2_bic_line_edit = QtWidgets.QLineEdit(self.tab2_est)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab2_bic_line_edit.sizePolicy().hasHeightForWidth())
        self.tab2_bic_line_edit.setSizePolicy(sizePolicy)
        self.tab2_bic_line_edit.setObjectName("tab2_bic_line_edit")
        self.tab2_bic_line_edit.setReadOnly(True)
        self.tab2_form.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.tab2_bic_line_edit)

        self.tab2_aic_label = QtWidgets.QLabel(self.tab2_est)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab2_aic_label.sizePolicy().hasHeightForWidth())
        self.tab2_aic_label.setSizePolicy(sizePolicy)
        self.tab2_aic_label.setObjectName("tab2_aic_label")
        self.tab2_form.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.tab2_aic_label)
        self.tab2_bic_label = QtWidgets.QLabel(self.tab2_est)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab2_bic_label.sizePolicy().hasHeightForWidth())
        self.tab2_bic_label.setSizePolicy(sizePolicy)
        self.tab2_bic_label.setObjectName("tab2_bic_label")
        self.tab2_form.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.tab2_bic_label)
        self.tab2_sse_label = QtWidgets.QLabel(self.tab2_est)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab2_sse_label.sizePolicy().hasHeightForWidth())
        self.tab2_sse_label.setSizePolicy(sizePolicy)
        self.tab2_sse_label.setObjectName("tab2_sse_label")
        self.tab2_form.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.tab2_sse_label)

        # SSE line edit
        self.tab2_sse_line_edit = QtWidgets.QLineEdit(self.tab2_est)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab2_sse_line_edit.sizePolicy().hasHeightForWidth())
        self.tab2_sse_line_edit.setSizePolicy(sizePolicy)
        self.tab2_sse_line_edit.setObjectName("tab2_sse_line_edit")
        self.tab2_sse_line_edit.setReadOnly(True)
        self.tab2_form.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.tab2_sse_line_edit)

        self.tab2_vertical.addLayout(self.tab2_form)
        self.tab2_srm_label = QtWidgets.QLabel(self.tab2_est)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab2_srm_label.sizePolicy().hasHeightForWidth())
        self.tab2_srm_label.setSizePolicy(sizePolicy)
        self.tab2_srm_label.setObjectName("tab2_srm_label")
        self.tab2_vertical.addWidget(self.tab2_srm_label)

        # radio buttons
        self.tab2_geometric_radio_button = QtWidgets.QRadioButton(self.tab2_est)
        self.tab2_geometric_radio_button.setObjectName("tab2_geometric_radio_button")
        self.tab2_geometric_radio_button.clicked.connect(self.est_geometric_SRM_slot)
        self.tab2_vertical.addWidget(self.tab2_geometric_radio_button)

        self.tab2_neg_binomial_2_radio_button = QtWidgets.QRadioButton(self.tab2_est)
        self.tab2_neg_binomial_2_radio_button.setObjectName("tab2_neg_binomial_2_radio_button")
        self.tab2_neg_binomial_2_radio_button.clicked.connect(self.est_nb2_SRM_slot)
        self.tab2_vertical.addWidget(self.tab2_neg_binomial_2_radio_button)

        self.tab2_weibull_2_radio_button = QtWidgets.QRadioButton(self.tab2_est)
        self.tab2_weibull_2_radio_button.setObjectName("tab2_weibull_2_radio_button")
        self.tab2_weibull_2_radio_button.clicked.connect(self.est_dw2_SRM_slot)
        self.tab2_vertical.addWidget(self.tab2_weibull_2_radio_button)

        '''
        self.tab2_neg_binomial_radio_button = QtWidgets.QRadioButton(self.tab2_est)
        self.tab2_neg_binomial_radio_button.setObjectName("tab2_neg_binomial_radio_button")
        self.tab2_neg_binomial_radio_button.clicked.connect(self.est_nb_SRM_slot)
        self.tab2_vertical.addWidget(self.tab2_neg_binomial_radio_button)

        self.tab2_weibull_radio_button = QtWidgets.QRadioButton(self.tab2_est)
        self.tab2_weibull_radio_button.setObjectName("tab2_weibull_radio_button")
        self.tab2_weibull_radio_button.clicked.connect(self.est_dw_SRM_slot)
        self.tab2_vertical.addWidget(self.tab2_weibull_radio_button)
        '''

        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.tab2_vertical.addItem(spacerItem)
        self.tab2_estimation_button = QtWidgets.QPushButton(self.tab2_est)
        self.tab2_estimation_button.setObjectName("tab2_estimation_button")
        self.tab2_vertical.addWidget(self.tab2_estimation_button, 0, QtCore.Qt.AlignHCenter)

        # self.tab2_estimation_button.clicked.connect(self.open_estimation_window)
        self.tab2_estimation_button.clicked.connect(self.run_estimation)

        # graphs
        self.tab2_layout.addLayout(self.tab2_vertical, 0, 2, 4, 1)
        # self.graphicsView_8 = QtWidgets.QGraphicsView(self.tab2_est)
        self.est_graph = PlotCanvas(width=5, height=4)
        self.est_intensity_graph = PlotCanvas(width=5, height=4)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.est_graph.sizePolicy().hasHeightForWidth())
        self.est_graph.setSizePolicy(sizePolicy)
        self.est_graph.setObjectName("est_graph")

        # graph tabs
        self.est_graph_tabs = QtWidgets.QTabWidget()
        self.est_graph_tabs.setAutoFillBackground(False)
        self.est_graph_tabs.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.est_graph_tabs.setUsesScrollButtons(False)
        self.est_graph_tabs.setObjectName("est_graph_tabs")

        self.est_graph_mvf_tab = QtWidgets.QWidget()
        self.est_graph_mvf_tab.setObjectName("est_graph_mvf_tab")

        self.est_graph_intensity_tab = QtWidgets.QWidget()
        self.est_graph_intensity_tab.setObjectName("est_graph_intensity_tab")

        self.est_graph_tabs.addTab(self.est_graph_mvf_tab, "")
        self.est_graph_tabs.addTab(self.est_graph_intensity_tab, "")

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)    # want graph and table to both stretch
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.est_graph_tabs.sizePolicy().hasHeightForWidth())
        self.est_graph_tabs.setSizePolicy(sizePolicy)

        self.est_mvf_layout = QtWidgets.QHBoxLayout(self.est_graph_mvf_tab)
        self.est_intensity_layout = QtWidgets.QHBoxLayout(self.est_graph_intensity_tab)

        self.est_mvf_layout.addWidget(self.est_graph)
        self.est_intensity_layout.addWidget(self.est_intensity_graph)

        self.tab2_layout.addWidget(self.est_graph_tabs, 1, 0, 5, 1)
        self.tabWidget.addTab(self.tab2_est, "")

    def init_tab3(self):
        self.tab3 = QtWidgets.QWidget()
        self.tab3.setObjectName("tab3")
        self.tabWidget.addTab(self.tab3, "")

    def init_tab4(self):
        self.tab4 = QtWidgets.QWidget()
        self.tab4.setObjectName("tab4")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab4)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_39 = QtWidgets.QLabel(self.tab4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_39.sizePolicy().hasHeightForWidth())
        self.label_39.setSizePolicy(sizePolicy)
        self.label_39.setObjectName("label_39")
        self.gridLayout_3.addWidget(self.label_39, 0, 0, 1, 1)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.label_37 = QtWidgets.QLabel(self.tab4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_37.sizePolicy().hasHeightForWidth())
        self.label_37.setSizePolicy(sizePolicy)
        self.label_37.setObjectName("label_37")
        self.verticalLayout_12.addWidget(self.label_37)
        self.checkBox_6 = QtWidgets.QCheckBox(self.tab4)
        self.checkBox_6.setObjectName("checkBox_6")
        self.verticalLayout_12.addWidget(self.checkBox_6)
        self.checkBox_7 = QtWidgets.QCheckBox(self.tab4)
        self.checkBox_7.setObjectName("checkBox_7")
        self.verticalLayout_12.addWidget(self.checkBox_7)
        self.checkBox_8 = QtWidgets.QCheckBox(self.tab4)
        self.checkBox_8.setObjectName("checkBox_8")
        self.verticalLayout_12.addWidget(self.checkBox_8)
        self.checkBox_9 = QtWidgets.QCheckBox(self.tab4)
        self.checkBox_9.setObjectName("checkBox_9")
        self.verticalLayout_12.addWidget(self.checkBox_9)
        self.checkBox_10 = QtWidgets.QCheckBox(self.tab4)
        self.checkBox_10.setObjectName("checkBox_10")
        self.verticalLayout_12.addWidget(self.checkBox_10)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_12.addItem(spacerItem1)
        self.label_38 = QtWidgets.QLabel(self.tab4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_38.sizePolicy().hasHeightForWidth())
        self.label_38.setSizePolicy(sizePolicy)
        self.label_38.setObjectName("label_38")
        self.verticalLayout_12.addWidget(self.label_38)
        self.radioButton_12 = QtWidgets.QRadioButton(self.tab4)
        self.radioButton_12.setObjectName("radioButton_12")
        self.verticalLayout_12.addWidget(self.radioButton_12)
        self.radioButton_13 = QtWidgets.QRadioButton(self.tab4)
        self.radioButton_13.setObjectName("radioButton_13")
        self.verticalLayout_12.addWidget(self.radioButton_13)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_12.addItem(spacerItem2)
        self.gridLayout_3.addLayout(self.verticalLayout_12, 0, 1, 2, 1)
        # self.graphicsView_9 = QtWidgets.QGraphicsView(self.tab4)
        self.rel_graph = PlotCanvas(width=5, height=4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.rel_graph.sizePolicy().hasHeightForWidth())
        self.rel_graph.setSizePolicy(sizePolicy)
        self.rel_graph.setObjectName("graphicsView_9")
        self.gridLayout_3.addWidget(self.rel_graph, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab4, "")

    def init_tab5(self):
        self.tab5 = QtWidgets.QWidget()
        self.tab5.setObjectName("tab5")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.tab5)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_27 = QtWidgets.QLabel(self.tab5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_27.sizePolicy().hasHeightForWidth())
        self.label_27.setSizePolicy(sizePolicy)
        self.label_27.setObjectName("label_27")
        self.gridLayout_6.addWidget(self.label_27, 0, 0, 1, 1)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_28 = QtWidgets.QLabel(self.tab5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_28.sizePolicy().hasHeightForWidth())
        self.label_28.setSizePolicy(sizePolicy)
        self.label_28.setObjectName("label_28")
        self.verticalLayout_10.addWidget(self.label_28)
        self.checkBox = QtWidgets.QCheckBox(self.tab5)
        self.checkBox.setObjectName("checkBox")
        self.verticalLayout_10.addWidget(self.checkBox)
        self.checkBox_2 = QtWidgets.QCheckBox(self.tab5)
        self.checkBox_2.setObjectName("checkBox_2")
        self.verticalLayout_10.addWidget(self.checkBox_2)
        self.checkBox_3 = QtWidgets.QCheckBox(self.tab5)
        self.checkBox_3.setObjectName("checkBox_3")
        self.verticalLayout_10.addWidget(self.checkBox_3)
        self.checkBox_4 = QtWidgets.QCheckBox(self.tab5)
        self.checkBox_4.setObjectName("checkBox_4")
        self.verticalLayout_10.addWidget(self.checkBox_4)
        self.checkBox_5 = QtWidgets.QCheckBox(self.tab5)
        self.checkBox_5.setObjectName("checkBox_5")
        self.verticalLayout_10.addWidget(self.checkBox_5)
        spacerItem3 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_10.addItem(spacerItem3)
        self.label_29 = QtWidgets.QLabel(self.tab5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_29.sizePolicy().hasHeightForWidth())
        self.label_29.setSizePolicy(sizePolicy)
        self.label_29.setObjectName("label_29")
        self.verticalLayout_10.addWidget(self.label_29)
        self.radioButton_5 = QtWidgets.QRadioButton(self.tab5)
        self.radioButton_5.setObjectName("radioButton_5")
        self.verticalLayout_10.addWidget(self.radioButton_5)
        self.radioButton_6 = QtWidgets.QRadioButton(self.tab5)
        self.radioButton_6.setObjectName("radioButton_6")
        self.verticalLayout_10.addWidget(self.radioButton_6)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_10.addItem(spacerItem4)
        self.gridLayout_6.addLayout(self.verticalLayout_10, 0, 1, 2, 1)
        self.graphicsView_7 = QtWidgets.QGraphicsView(self.tab5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.graphicsView_7.sizePolicy().hasHeightForWidth())
        self.graphicsView_7.setSizePolicy(sizePolicy)
        self.graphicsView_7.setObjectName("graphicsView_7")
        self.gridLayout_6.addWidget(self.graphicsView_7, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab5, "")

    def init_tab6(self):
        self.tab6 = QtWidgets.QWidget()
        self.tab6.setObjectName("tab6")
        self.tabWidget.addTab(self.tab6, "")

    def init_menubar(self):
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1401, 26))
        self.menubar.setObjectName("menubar")
        self.menuf = QtWidgets.QMenu(self.menubar)
        self.menuf.setObjectName("menuf")
        self.menuSettings = QtWidgets.QMenu(self.menubar)
        self.menuSettings.setObjectName("menuSettings")
        MainWindow.setMenuBar(self.menubar)

        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setShortcut("Ctrl+O")
        self.actionOpen.setStatusTip("Load failure data from a file")
        self.actionOpen.triggered.connect(self.loadCsv)
        # self.actionOpen.triggered.connect(self.tableWidget.loadCsv)

        self.actionOpen.setObjectName("actionOpen")
        self.menuf.addAction(self.actionOpen)
        self.menubar.addAction(self.menuf.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())
    
    '''
    def open_estimation_window(self):
        dialog = QtWidgets.QDialog()
        est_window = estimation_setting.Ui_Dialog()
        est_window.setupUi(dialog)
        #self.est_window.setModal()
        dialog.exec_()
    '''
    
    def loadCsv(self, fileName):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open CSV",
                (QtCore.QDir.homePath()), "CSV (*.csv *.tsv)")
        if fileName:
            #print(fileName)
            self.file_name.setText(fileName)
            ff = open(fileName, 'r')
            mytext = ff.read()
            # print(mytext)
            ff.close()
            f = open(fileName, 'r')
            with f:
                self.fname = os.path.splitext(str(fileName))[0].split("/")[-1]
            #    self.setWindowTitle(self.fname)
                reader = csv.reader(f, delimiter = ',')
                self.model.clear()
                gv.raw_imported_data = []
                for row in reader:
                    items = [QtGui.QStandardItem(field) for field in row]
                    gv.raw_imported_data.append(row)
                    self.model.appendRow(items)
                self.tableView.resizeColumnsToContents()

                # store data in covariate file
                covariate.initialization()

                # ALL MATH, GRAPHS HAPPEN AFTER LOADING FILE
                self.l.loaded.emit()  # emit loaded signal
                
                # plot fault graph
                self.fault_graph.fault_plot()
                self.fault_intensity_graph.fault_intensity_plot()

                #print(raw_imported_data)
            #    if mytext.count(';') <= mytext.count('\t'):
            #        reader = csv.reader(f, delimiter = '\t')
            #        self.model.clear()
            #        for row in reader:    
            #            items = [QtGui.QStandardItem(field) for field in row]
            #            self.model.appendRow(items)
            #        self.tableView.resizeColumnsToContents()
            #    else:
            #        reader = csv.reader(f, delimiter = ';')
            #        self.model.clear()
            #        for row in reader:    
            #            items = [QtGui.QStandardItem(field) for field in row]
            #            self.model.appendRow(items)
            #        self.tableView.resizeColumnsToContents()

    def cv_plotting(self):
        if (gv.has_data):
            print("*** Starting estimation thread... ***")
            self.est_graph.loading_plot()
            self.est_intensity_graph.loading_plot()
            covariate.main()
            gv.estimation_ran = True    # set flag that says estimation step has been run

            # self.fault_graph.fault_plot()
            # self.est_graph.estimation_plot()
            self.est_geometric_SRM_slot()                       # geometric SRM shown by default
            self.tab2_geometric_radio_button.setChecked(True)   # geometric SRM chosen by default
            self.rel_graph.reliability_plot()
            
            self.tab2_llf_line_edit.setText("{:0.5f}".format(gv.llf_val))
            self.tab2_aic_line_edit.setText("{:0.5f}".format(gv.aic_val))
            self.tab2_bic_line_edit.setText("{:0.5f}".format(gv.bic_val))
            self.tab2_sse_line_edit.setText("{:0.5f}".format(gv.sse_val))

            
            print("*** Estimation complete. ***")

    def run_estimation(self):
        cv_thread = threading.Thread(target = self.cv_plotting)
        cv_thread.start()

    def data_loaded_slot(self):
        gv.has_data = True      # set flag to say that data has been loaded

    def est_geometric_SRM_slot(self):
        if (gv.has_data and gv.estimation_ran):
            covariate.model_fitting("geometric")
            self.est_graph.estimation_plot()                # plot tabs on both plots
            self.est_intensity_graph.estimation_intensity_plot()
            self.tab2_llf_line_edit.setText("{:0.5f}".format(gv.llf_val))
            self.tab2_aic_line_edit.setText("{:0.5f}".format(gv.aic_val))
            self.tab2_bic_line_edit.setText("{:0.5f}".format(gv.bic_val))
            self.tab2_sse_line_edit.setText("{:0.5f}".format(gv.sse_val))

    def est_nb2_SRM_slot(self):
        if (gv.has_data and gv.estimation_ran):
            covariate.model_fitting("nb2")
            self.est_graph.estimation_plot()
            self.est_intensity_graph.estimation_intensity_plot()
            self.tab2_llf_line_edit.setText("{:0.5f}".format(gv.llf_val))
            self.tab2_aic_line_edit.setText("{:0.5f}".format(gv.aic_val))
            self.tab2_bic_line_edit.setText("{:0.5f}".format(gv.bic_val))
            self.tab2_sse_line_edit.setText("{:0.5f}".format(gv.sse_val))

    def est_dw2_SRM_slot(self):
        if (gv.has_data and gv.estimation_ran):
            covariate.model_fitting("dw2")
            self.est_graph.estimation_plot()
            self.est_intensity_graph.estimation_intensity_plot()
            self.tab2_llf_line_edit.setText("{:0.5f}".format(gv.llf_val))
            self.tab2_aic_line_edit.setText("{:0.5f}".format(gv.aic_val))
            self.tab2_bic_line_edit.setText("{:0.5f}".format(gv.bic_val))
            self.tab2_sse_line_edit.setText("{:0.5f}".format(gv.sse_val))
    
    '''
    def est_nb_SRM_slot(self):
        if (gv.has_data):
            covariate.model_fitting("nb")
            self.est_graph.estimation_plot()
            self.tab2_llf_line_edit.setText("{:0.5f}".format(gv.llf_val))
            self.tab2_aic_line_edit.setText("{:0.5f}".format(gv.aic_val))
            self.tab2_bic_line_edit.setText("{:0.5f}".format(gv.bic_val))
            self.tab2_sse_line_edit.setText("{:0.5f}".format(gv.sse_val))

    def est_dw_SRM_slot(self):
        if (gv.has_data):
            covariate.model_fitting("dw")
            self.est_graph.estimation_plot()
            self.tab2_llf_line_edit.setText("{:0.5f}".format(gv.llf_val))
            self.tab2_aic_line_edit.setText("{:0.5f}".format(gv.aic_val))
            self.tab2_bic_line_edit.setText("{:0.5f}".format(gv.bic_val))
            self.tab2_sse_line_edit.setText("{:0.5f}".format(gv.sse_val))
    '''

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.table_area_label.setText(_translate("MainWindow", "Imported data table"))
        self.fault_graph_label.setText(_translate("MainWindow", "Imported data graph"))
        self.selected_file_label.setText(_translate("MainWindow", "Selected file:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab1_fault), _translate("MainWindow", "Fault data"))
        self.fault_graph_tabs.setTabText(self.fault_graph_tabs.indexOf(self.fault_graph_mvf_tab), _translate("MainWindow", "MVF"))   # MainWindow?
        self.fault_graph_tabs.setTabText(self.fault_graph_tabs.indexOf(self.fault_graph_intensity_tab), _translate("MainWindow", "Intensity"))   # MainWindow?
        # self.est_graph_label.setText(_translate("MainWindow", "Graph area"))
        # self.est_gof_label.setText(_translate("MainWindow", "Goodness-of-fit measures"))
        # self.est_LLF_label.setText(_translate("MainWindow", "LLF"))
        # self.est_AIC_label.setText(_translate("MainWindow", "AIC"))
        # self.est_BIC_label.setText(_translate("MainWindow", "BIC"))
        # self.est_MSE_label.setText(_translate("MainWindow", "MSE"))
        # self.est_SRMs_label.setText(_translate("MainWindow", "Selected SRMs"))
        # __sortingEnabled = self.est_list.isSortingEnabled()
        # self.est_list.setSortingEnabled(False)
        # item = self.est_list.item(0)
        # item.setText(_translate("MainWindow", "Geometric"))
        # item = self.est_list.item(1)
        # item.setText(_translate("MainWindow", "Discrete Weibull"))
        # item = self.est_list.item(2)
        # item.setText(_translate("MainWindow", "Discrete Weibull (order 2)"))
        # item = self.est_list.item(3)
        # item.setText(_translate("MainWindow", "Negative binomial"))
        # item = self.est_list.item(4)
        # item.setText(_translate("MainWindow", "Negative binomial (order 2)"))
        # self.est_list.setSortingEnabled(__sortingEnabled)
        # self.est_estimation_button.setText(_translate("MainWindow", "Estimation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab2_est), _translate("MainWindow", "Estimation"))
        self.est_graph_tabs.setTabText(self.est_graph_tabs.indexOf(self.est_graph_mvf_tab), _translate("MainWindow", "MVF"))   # MainWindow?
        self.est_graph_tabs.setTabText(self.est_graph_tabs.indexOf(self.est_graph_intensity_tab), _translate("MainWindow", "Intensity"))   # MainWindow?
        self.tab2_graph_label.setText(_translate("MainWindow", "Estimation graph"))
        self.tab2_gof_label.setText(_translate("MainWindow", "Goodness-of-fit measures"))
        self.tab2_llf_label.setText(_translate("MainWindow", "LLF"))
        self.tab2_aic_label.setText(_translate("MainWindow", "AIC"))
        self.tab2_bic_label.setText(_translate("MainWindow", "BIC"))
        self.tab2_sse_label.setText(_translate("MainWindow", "SSE"))
        self.tab2_srm_label.setText(_translate("MainWindow", "Selected SRMs"))
        self.tab2_geometric_radio_button.setText(_translate("MainWindow", "Geometric"))
        self.tab2_neg_binomial_2_radio_button.setText(_translate("MainWindow", "Negative binomial (order 2)"))
        self.tab2_weibull_2_radio_button.setText(_translate("MainWindow", "Discrete Weibull (order 2)"))
        # self.tab2_neg_binomial_radio_button.setText(_translate("MainWindow", "Negative binomial"))
        # self.tab2_weibull_radio_button.setText(_translate("MainWindow", "Discrete Weibull"))
        self.tab2_estimation_button.setText(_translate("MainWindow", "Start estimation"))
        # self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab3), _translate("MainWindow", "Statistical test"))
        # self.label_39.setText(_translate("MainWindow", "Graph area"))
        # self.label_37.setText(_translate("MainWindow", "Selected SRMs"))
        # self.checkBox_6.setText(_translate("MainWindow", "Geometric"))
        # self.checkBox_7.setText(_translate("MainWindow", "Negative binomial (order 2)"))
        # self.checkBox_8.setText(_translate("MainWindow", "Discrete Weibull (order 2)"))
        # self.checkBox_9.setText(_translate("MainWindow", "Negative binomial"))
        # self.checkBox_10.setText(_translate("MainWindow", "Discrete Weibull"))
        # self.label_38.setText(_translate("MainWindow", "Graphical display"))
        # self.radioButton_12.setText(_translate("MainWindow", "Mean value function"))
        # self.radioButton_13.setText(_translate("MainWindow", "Intensity"))
        # self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab4), _translate("MainWindow", "Mean value/intensity"))
        self.label_27.setText(_translate("MainWindow", "Graph area"))
        self.label_28.setText(_translate("MainWindow", "Selected SRMs"))
        self.checkBox.setText(_translate("MainWindow", "Geometric"))
        self.checkBox_2.setText(_translate("MainWindow", "Negative binomial (order 2)"))
        self.checkBox_3.setText(_translate("MainWindow", "Discrete Weibull (order 2)"))
        self.checkBox_4.setText(_translate("MainWindow", "Negative binomial"))
        self.checkBox_5.setText(_translate("MainWindow", "Discrete Weibull"))
        self.label_29.setText(_translate("MainWindow", "Graphical display"))
        self.radioButton_5.setText(_translate("MainWindow", "Mean value function"))
        self.radioButton_6.setText(_translate("MainWindow", "Intensity"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab5), _translate("MainWindow", "Reliability/prediction"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab6), _translate("MainWindow", "Optimal effort allocation"))
        self.menuf.setTitle(_translate("MainWindow", "File"))
        self.menuSettings.setTitle(_translate("MainWindow", "Settings"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # self.plot()

    def fault_plot(self):
        data_y = gv.kVec_cumulative
        data_x = gv.failure_times
        self.fault_fig = self.figure.add_subplot(111)
        self.fault_step, = self.fault_fig.step(data_x, data_y, 'b', where="post")        # step function
        # self.failure_step.set_alpha(0)
        # print(self.failure_step.get_alpha())
        # self.fitted_plot, = self.ax.plot(data_x, gv.mvf_list, 'ro')
        self.fault_fig.set_xlabel("calendar time")
        self.fault_fig.set_ylabel("number of failures")
        self.fault_fig.grid(True)
        #ax.set_title('PyQt Matplotlib Example')
        self.draw()     # re-draws figurem allows graph to change
        self.show()     # displays current figure

        # self.plot_on(self.failure_step)
        # self.plot_off(self.fitted_plot)

    def fault_intensity_plot(self):
        # histogram
        x = gv.failure_times
        y = gv.kVec
        self.fault_intensity_fig = self.figure.add_subplot(111)
        self.fault_hist = self.fault_intensity_fig.bar(x, height=y)
        self.fault_intensity_fig.set_xlabel("time")
        self.fault_intensity_fig.set_ylabel("intensity")
        self.fault_intensity_fig.grid(True)
        self.draw()
        self.show()

    def estimation_plot(self):
        data_y = gv.kVec_cumulative
        data_x = gv.failure_times
        self.est_fig = self.figure.add_subplot(111)
        self.est_fig.clear()         # clear old plot before new one is drawn
        self.est_step, = self.est_fig.step(data_x, data_y, 'b', where="post", label="imported fault data")        # step function
        # self.failure_step.set_alpha(0)
        # print(self.failure_step.get_alpha())
        self.geometric_plot, = self.est_fig.plot(data_x, gv.mvf_list, 'ro-', label="fitted data")
        self.est_fig.set_ylim([0, None])
        self.est_fig.set_xlabel("calendar time")
        self.est_fig.set_ylabel("number of failures")
        self.est_fig.grid(True)
        self.est_fig.legend(loc="upper left")
        #ax.set_title('PyQt Matplotlib Example')
        self.draw()     # re-draws figurem allows graph to change
        self.show()     # displays current figure
        self.flush_events()

    def estimation_intensity_plot(self):
        # histogram
        x = gv.failure_times
        y = gv.kVec
        self.est_intensity_fig = self.figure.add_subplot(111)
        self.est_intensity_fig.clear()
        self.est_hist = self.est_intensity_fig.bar(x, height=y, label="imported fault data")
        
        self.est_intensity_plot, = self.est_intensity_fig.plot(x, gv.intensity_list, 'ro-', label="fitted data")
        self.est_intensity_fig.set_xlabel("time")
        self.est_intensity_fig.set_ylabel("intensity")
        self.est_intensity_fig.grid(True)
        self.est_fig.legend(loc="upper right")
        self.draw()
        self.show()
        self.flush_events()

    def reliability_plot(self):
        # histogram
        x = gv.failure_times
        y = gv.kVec
        self.rel_fig = self.figure.add_subplot(111)
        self.rel_hist = self.rel_fig.bar(x, height=y)
        self.rel_fig.set_xlabel("time")
        self.rel_fig.set_ylabel("intensity")
        self.rel_fig.grid()
        self.draw()
        self.show()

    def loading_plot(self):
        self.est_fig = self.figure.add_subplot(111)
        self.est_fig.clear()         # clear old plot before new one is drawn
        self.est_loading = self.est_fig.text(0.5, 0.5, "Running estimation...", horizontalalignment="center", verticalalignment="center")        # text function
        self.draw()     # re-draws figurem allows graph to change
        self.show()     # displays current figure
        self.flush_events()

    def plot_on(self, plt):
        # self.ax.set_visible(not self.ax.get_visible())
        plt.set_alpha(1.0)
        # self.draw()

    def plot_off(self, plt):
        plt.set_alpha(0.0)

class DataLoaded(QtCore.QObject):
    loaded = QtCore.pyqtSignal()
    # update line edits and stuff
    #   LLF, AIC, BIC, MSE
    # file label?
    # do the plotting

class DataTable(QtWidgets.QTableView):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        pass
    
    def loadCsv(self, fileName):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV",
                (QtCore.QDir.homePath()), "CSV (*.csv *.tsv)")
    
        if fileName:
            print(fileName)
            ff = open(fileName, 'r')
            mytext = ff.read()
            # print(mytext)
            ff.close()
            f = open(fileName, 'r')
            with f:
                self.fname = os.path.splitext(str(fileName))[0].split("/")[-1]
                self.setWindowTitle(self.fname)
                if mytext.count(';') <= mytext.count('\t'):
                    reader = csv.reader(f, delimiter = '\t')
                    # self.model.clear()
                    for row in reader:    
                        items = [QtGui.QStandardItem(field) for field in row]
                        Ui_MainWindow.model.appendRow(items)
                    self.tableView.resizeColumnsToContents()
                else:
                    reader = csv.reader(f, delimiter = ';')
                    # self.model.clear()
                    for row in reader:    
                        items = [QtGui.QStandardItem(field) for field in row]
                        Ui_MainWindow.model.appendRow(items)
                    self.tableView.resizeColumnsToContents()

    '''
    def open_sheet(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV", os.getenv("HOME"), "CSV(*.csv)")
        if path[0] != '':
            with open(path[0], newline='') as csv_file:
                self.setRowCount()
    '''

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())