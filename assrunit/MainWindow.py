import sys
import string

from PySide6 import QtCore
from matplotlib.table import table

from ui.assr_gui import *
#from PySide6.QtWidgets import *
#from PySide6.QtGui import *
#from PySide6.QtCore import *


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.assign_widgets()
        # show window
        self.show()

    def assign_widgets(self):
        # close button
        self.actionClose.triggered.connect(self.close)

        # create problems TODO: should be in a file or somewhere else to read from
        problem_entries = ['please choose...', 'shizophrenia', 'bipolar disorder', 'autism', 'epilepsy']
        self.comboBox.addItems(problem_entries)
        self.comboBox.currentIndexChanged.connect(self.change_label_comboBox2)

        # model type checkbox
        self.comboBox_2.addItem('please choose the problem first...')

        # fill parameter table

        # create entries TODO: should be read from module files
        table_parameters = ['please choose type of models first']

        self.comboBox_2.currentIndexChanged.connect(self.fill_param_table)

        self.listWidget.addItems(table_parameters)

        model_entries = ['please choose parameters first...']
        self.listWidget_2.addItems(model_entries)

        self.confirmButton.clicked.connect(self.fill_compatible_models)

    def change_label_comboBox2(self):

        models_label_text = str(self.comboBox.currentText())

        self.label_2.setText(models_label_text+' models:')
        self.label_2.repaint()
        #TODO gray out option 0 + create model type should be in a file or somewhere else to read from

        #TODO doesn't work
        self.comboBox_2.setItemText(0,'please choose ...')
        # update models combo box
        model_entries = ['auditory models', 'other models']
        if self.comboBox_2.count() < 2:
            self.comboBox_2.addItems(model_entries)

    def fill_compatible_models(self):
        model_entries = [ 'vierling', 'chandelier', 'other model']

        # update model list
        self.listWidget_2.clear()
        self.listWidget_2.addItems(model_entries)

    def fill_param_table(self):
        # update models list
        self.listWidget.clear()

        table_parameters = []
        for ch in string.ascii_lowercase:
            table_parameters.append(ch + '         ')

        self.listWidget.addItems(table_parameters)


if __name__ == '__main__':
    # create instance of application and window
    app = QApplication(sys.argv)
    mainWin = MainWindow()

    # run the application
    ret = app.exec_()

    sys.exit(ret)
