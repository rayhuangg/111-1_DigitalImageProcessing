# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 606)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tabWidget.setElideMode(QtCore.Qt.ElideLeft)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.button_file = QtWidgets.QPushButton(self.tab)
        self.button_file.setGeometry(QtCore.QRect(90, 40, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file.setFont(font)
        self.button_file.setObjectName("button_file")
        self.label_img_1 = QtWidgets.QLabel(self.tab)
        self.label_img_1.setGeometry(QtCore.QRect(90, 160, 200, 150))
        self.label_img_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_1.setObjectName("label_img_1")
        self.label_img_2 = QtWidgets.QLabel(self.tab)
        self.label_img_2.setGeometry(QtCore.QRect(440, 160, 200, 150))
        self.label_img_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_2.setObjectName("label_img_2")
        self.comboBox_color_model = QtWidgets.QComboBox(self.tab)
        self.comboBox_color_model.setGeometry(QtCore.QRect(290, 40, 201, 31))
        self.comboBox_color_model.setObjectName("comboBox_color_model")
        self.comboBox_color_model.addItem("")
        self.comboBox_color_model.addItem("")
        self.comboBox_color_model.addItem("")
        self.comboBox_color_model.addItem("")
        self.comboBox_color_model.addItem("")
        self.comboBox_color_model.addItem("")
        self.textEdit_model_type = QtWidgets.QTextEdit(self.tab)
        self.textEdit_model_type.setGeometry(QtCore.QRect(90, 330, 201, 31))
        self.textEdit_model_type.setObjectName("textEdit_model_type")
        self.textEdit_model_type_2 = QtWidgets.QTextEdit(self.tab)
        self.textEdit_model_type_2.setGeometry(QtCore.QRect(450, 330, 201, 31))
        self.textEdit_model_type_2.setObjectName("textEdit_model_type_2")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_img_4 = QtWidgets.QLabel(self.tab_2)
        self.label_img_4.setGeometry(QtCore.QRect(330, 250, 200, 150))
        self.label_img_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_4.setObjectName("label_img_4")
        self.label_threshold_2 = QtWidgets.QLabel(self.tab_2)
        self.label_threshold_2.setGeometry(QtCore.QRect(70, 180, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_2.setFont(font)
        self.label_threshold_2.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_2.setObjectName("label_threshold_2")
        self.label_img_3 = QtWidgets.QLabel(self.tab_2)
        self.label_img_3.setGeometry(QtCore.QRect(70, 250, 200, 150))
        self.label_img_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_3.setObjectName("label_img_3")
        self.label_threshold_3 = QtWidgets.QLabel(self.tab_2)
        self.label_threshold_3.setGeometry(QtCore.QRect(330, 180, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_3.setFont(font)
        self.label_threshold_3.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_3.setObjectName("label_threshold_3")
        self.button_file_2 = QtWidgets.QPushButton(self.tab_2)
        self.button_file_2.setGeometry(QtCore.QRect(20, 50, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file_2.setFont(font)
        self.button_file_2.setObjectName("button_file_2")
        self.label_cutoff = QtWidgets.QLabel(self.tab_2)
        self.label_cutoff.setGeometry(QtCore.QRect(370, 100, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_cutoff.setFont(font)
        self.label_cutoff.setObjectName("label_cutoff")
        self.horizontalSlider_cutoff = QtWidgets.QSlider(self.tab_2)
        self.horizontalSlider_cutoff.setGeometry(QtCore.QRect(190, 100, 160, 22))
        self.horizontalSlider_cutoff.setMinimum(1)
        self.horizontalSlider_cutoff.setMaximum(50)
        self.horizontalSlider_cutoff.setProperty("value", 1)
        self.horizontalSlider_cutoff.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_cutoff.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider_cutoff.setTickInterval(5)
        self.horizontalSlider_cutoff.setObjectName("horizontalSlider_cutoff")
        self.label_threshold_9 = QtWidgets.QLabel(self.tab_2)
        self.label_threshold_9.setGeometry(QtCore.QRect(170, 40, 231, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_9.setFont(font)
        self.label_threshold_9.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_9.setObjectName("label_threshold_9")
        self.button_go = QtWidgets.QPushButton(self.tab_2)
        self.button_go.setGeometry(QtCore.QRect(680, 70, 51, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_go.setFont(font)
        self.button_go.setObjectName("button_go")
        self.label_threshold_11 = QtWidgets.QLabel(self.tab_2)
        self.label_threshold_11.setGeometry(QtCore.QRect(650, 40, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_11.setFont(font)
        self.label_threshold_11.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_11.setObjectName("label_threshold_11")
        self.label_threshold_14 = QtWidgets.QLabel(self.tab_2)
        self.label_threshold_14.setGeometry(QtCore.QRect(590, 180, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_14.setFont(font)
        self.label_threshold_14.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_14.setObjectName("label_threshold_14")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.button_file_3 = QtWidgets.QPushButton(self.tab_3)
        self.button_file_3.setGeometry(QtCore.QRect(30, 90, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file_3.setFont(font)
        self.button_file_3.setObjectName("button_file_3")
        self.label_threshold_4 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_4.setGeometry(QtCore.QRect(40, 190, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_4.setFont(font)
        self.label_threshold_4.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_4.setObjectName("label_threshold_4")
        self.label_brightness = QtWidgets.QLabel(self.tab_3)
        self.label_brightness.setGeometry(QtCore.QRect(220, 140, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_brightness.setFont(font)
        self.label_brightness.setText("")
        self.label_brightness.setObjectName("label_brightness")
        self.horizontalSlider_gh = QtWidgets.QSlider(self.tab_3)
        self.horizontalSlider_gh.setGeometry(QtCore.QRect(40, 250, 160, 22))
        self.horizontalSlider_gh.setMinimum(0)
        self.horizontalSlider_gh.setMaximum(20)
        self.horizontalSlider_gh.setPageStep(5)
        self.horizontalSlider_gh.setProperty("value", 20)
        self.horizontalSlider_gh.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_gh.setObjectName("horizontalSlider_gh")
        self.label_gh = QtWidgets.QLabel(self.tab_3)
        self.label_gh.setGeometry(QtCore.QRect(220, 250, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_gh.setFont(font)
        self.label_gh.setObjectName("label_gh")
        self.label_gl = QtWidgets.QLabel(self.tab_3)
        self.label_gl.setGeometry(QtCore.QRect(220, 360, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_gl.setFont(font)
        self.label_gl.setObjectName("label_gl")
        self.horizontalSlider_gl = QtWidgets.QSlider(self.tab_3)
        self.horizontalSlider_gl.setGeometry(QtCore.QRect(40, 360, 160, 22))
        self.horizontalSlider_gl.setMinimum(0)
        self.horizontalSlider_gl.setMaximum(20)
        self.horizontalSlider_gl.setPageStep(5)
        self.horizontalSlider_gl.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_gl.setObjectName("horizontalSlider_gl")
        self.label_threshold_5 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_5.setGeometry(QtCore.QRect(40, 300, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_5.setFont(font)
        self.label_threshold_5.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_5.setObjectName("label_threshold_5")
        self.label_threshold_7 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_7.setGeometry(QtCore.QRect(470, 30, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_7.setFont(font)
        self.label_threshold_7.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_7.setObjectName("label_threshold_7")
        self.label_img_5 = QtWidgets.QLabel(self.tab_3)
        self.label_img_5.setGeometry(QtCore.QRect(470, 80, 200, 150))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_img_5.setFont(font)
        self.label_img_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_5.setObjectName("label_img_5")
        self.label_img_6 = QtWidgets.QLabel(self.tab_3)
        self.label_img_6.setGeometry(QtCore.QRect(470, 310, 200, 150))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_img_6.setFont(font)
        self.label_img_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_6.setObjectName("label_img_6")
        self.label_threshold_8 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_8.setGeometry(QtCore.QRect(470, 260, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_8.setFont(font)
        self.label_threshold_8.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_8.setObjectName("label_threshold_8")
        self.label_threshold_15 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_15.setGeometry(QtCore.QRect(30, 20, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_15.setFont(font)
        self.label_threshold_15.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_15.setObjectName("label_threshold_15")
        self.tabWidget.addTab(self.tab_3, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_file.setText(_translate("MainWindow", "choose img"))
        self.label_img_1.setText(_translate("MainWindow", "img"))
        self.label_img_2.setText(_translate("MainWindow", "result"))
        self.comboBox_color_model.setItemText(0, _translate("MainWindow", "-- Choose Color Model --"))
        self.comboBox_color_model.setItemText(1, _translate("MainWindow", "CMY"))
        self.comboBox_color_model.setItemText(2, _translate("MainWindow", "HSI"))
        self.comboBox_color_model.setItemText(3, _translate("MainWindow", "XYZ"))
        self.comboBox_color_model.setItemText(4, _translate("MainWindow", "Lab"))
        self.comboBox_color_model.setItemText(5, _translate("MainWindow", "YUV"))
        self.textEdit_model_type.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">model: RGB</p></body></html>"))
        self.textEdit_model_type_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">model:</p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Part1"))
        self.label_img_4.setText(_translate("MainWindow", "result"))
        self.label_threshold_2.setText(_translate("MainWindow", "grayscale"))
        self.label_img_3.setText(_translate("MainWindow", "img"))
        self.label_threshold_3.setText(_translate("MainWindow", "pseudo-color"))
        self.button_file_2.setText(_translate("MainWindow", "choose img"))
        self.label_cutoff.setText(_translate("MainWindow", "value"))
        self.label_threshold_9.setText(_translate("MainWindow", "1. Cut-off Frequency"))
        self.button_go.setText(_translate("MainWindow", "GO"))
        self.label_threshold_11.setText(_translate("MainWindow", "3."))
        self.label_threshold_14.setText(_translate("MainWindow", "color bar"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Part2"))
        self.button_file_3.setText(_translate("MainWindow", "choose img"))
        self.label_threshold_4.setText(_translate("MainWindow", "Gamma_H"))
        self.label_gh.setText(_translate("MainWindow", "value"))
        self.label_gl.setText(_translate("MainWindow", "value"))
        self.label_threshold_5.setText(_translate("MainWindow", "Gamma_L"))
        self.label_threshold_7.setText(_translate("MainWindow", "Original"))
        self.label_img_5.setText(_translate("MainWindow", "img"))
        self.label_img_6.setText(_translate("MainWindow", "img"))
        self.label_threshold_8.setText(_translate("MainWindow", "homomorphic"))
        self.label_threshold_15.setText(_translate("MainWindow", "K-means"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Part3"))
