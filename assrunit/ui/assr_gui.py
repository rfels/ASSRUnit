# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'assr_gui.ui'
##
## Created by: Qt User Interface Compiler version 6.0.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 869)
        MainWindow.setStyleSheet(u"QGroupBox\n"
"{\n"
"    font-size: 15px;\n"
"    font-weight: bold;\n"
"}")
        self.actionClose = QAction(MainWindow)
        self.actionClose.setObjectName(u"actionClose")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(30, 0, 751, 156))
        self.groupBox.setStyleSheet(u"")
        self.groupBox.setAlignment(Qt.AlignHCenter|Qt.AlignTop)
        self.groupBox.setCheckable(False)
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(60, 40, 201, 20))
        self.comboBox = QComboBox(self.groupBox)
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(280, 40, 79, 26))
        self.comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(60, 90, 201, 18))
        self.comboBox_2 = QComboBox(self.groupBox)
        self.comboBox_2.setObjectName(u"comboBox_2")
        self.comboBox_2.setGeometry(QRect(280, 90, 79, 26))
        self.comboBox_2.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(30, 160, 751, 271))
        self.groupBox_2.setAlignment(Qt.AlignCenter)
        self.lineEdit = QLineEdit(self.groupBox_2)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(150, 30, 171, 26))
        self.selectAllButton = QPushButton(self.groupBox_2)
        self.selectAllButton.setObjectName(u"selectAllButton")
        self.selectAllButton.setGeometry(QRect(460, 30, 80, 26))
        self.selectNoneButton = QPushButton(self.groupBox_2)
        self.selectNoneButton.setObjectName(u"selectNoneButton")
        self.selectNoneButton.setGeometry(QRect(560, 30, 80, 26))
        self.label_3 = QLabel(self.groupBox_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(70, 30, 58, 18))
        self.listWidget = QListWidget(self.groupBox_2)
        self.listWidget.setObjectName(u"listWidget")
        self.listWidget.setGeometry(QRect(30, 70, 691, 192))
        self.listWidget.setAlternatingRowColors(True)
        self.listWidget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.listWidget.setFlow(QListView.LeftToRight)
        self.listWidget.setProperty("isWrapping", True)
        self.listWidget.setSelectionRectVisible(True)
        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(30, 520, 751, 271))
        self.groupBox_3.setAlignment(Qt.AlignCenter)
        self.listWidget_2 = QListWidget(self.groupBox_3)
        self.listWidget_2.setObjectName(u"listWidget_2")
        self.listWidget_2.setGeometry(QRect(240, 30, 256, 231))
        self.confirmButton = QPushButton(self.centralwidget)
        self.confirmButton.setObjectName(u"confirmButton")
        self.confirmButton.setGeometry(QRect(310, 480, 131, 26))
        self.changeButton = QPushButton(self.centralwidget)
        self.changeButton.setObjectName(u"changeButton")
        self.changeButton.setGeometry(QRect(310, 440, 131, 26))
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(100, 480, 191, 20))
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(100, 440, 181, 18))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 23))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.actionClose)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"ASSRUnit GUI", None))
        self.actionClose.setText(QCoreApplication.translate("MainWindow", u"Close", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Which  models do you want to analyse?", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"problem/desease/disorder:", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Type of models to choose:", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Please select the microscopic parameters of the models of interest", None))
        self.selectAllButton.setText(QCoreApplication.translate("MainWindow", u"select all", None))
        self.selectNoneButton.setText(QCoreApplication.translate("MainWindow", u"select none", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Search:", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Here are the models compatible to the chosen parameters. Choose Models to analyse and visualize.", None))
        self.confirmButton.setText(QCoreApplication.translate("MainWindow", u"confirm choice", None))
        self.changeButton.setText(QCoreApplication.translate("MainWindow", u"change values", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"confirm selected parameters:", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"change parameter values:", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
    # retranslateUi

