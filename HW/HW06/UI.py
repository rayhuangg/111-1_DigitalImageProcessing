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
        MainWindow.resize(801, 606)
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
        self.button_original = QtWidgets.QPushButton(self.tab)
        self.button_original.setGeometry(QtCore.QRect(10, 110, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_original.setFont(font)
        self.button_original.setObjectName("button_original")
        self.label_img_1 = QtWidgets.QLabel(self.tab)
        self.label_img_1.setGeometry(QtCore.QRect(160, 30, 200, 200))
        self.label_img_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_1.setObjectName("label_img_1")
        self.button_trapezoidal = QtWidgets.QPushButton(self.tab)
        self.button_trapezoidal.setGeometry(QtCore.QRect(390, 100, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_trapezoidal.setFont(font)
        self.button_trapezoidal.setObjectName("button_trapezoidal")
        self.button_wavy = QtWidgets.QPushButton(self.tab)
        self.button_wavy.setGeometry(QtCore.QRect(10, 400, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_wavy.setFont(font)
        self.button_wavy.setObjectName("button_wavy")
        self.button_circular = QtWidgets.QPushButton(self.tab)
        self.button_circular.setGeometry(QtCore.QRect(400, 410, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_circular.setFont(font)
        self.button_circular.setObjectName("button_circular")
        self.button_file_1 = QtWidgets.QPushButton(self.tab)
        self.button_file_1.setGeometry(QtCore.QRect(20, 20, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file_1.setFont(font)
        self.button_file_1.setObjectName("button_file_1")
        self.label_img_2 = QtWidgets.QLabel(self.tab)
        self.label_img_2.setGeometry(QtCore.QRect(550, 30, 200, 200))
        self.label_img_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_2.setObjectName("label_img_2")
        self.label_img_7 = QtWidgets.QLabel(self.tab)
        self.label_img_7.setGeometry(QtCore.QRect(160, 260, 200, 200))
        self.label_img_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_7.setObjectName("label_img_7")
        self.label_img_8 = QtWidgets.QLabel(self.tab)
        self.label_img_8.setGeometry(QtCore.QRect(550, 270, 200, 200))
        self.label_img_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_8.setObjectName("label_img_8")
        self.label_threshold_4 = QtWidgets.QLabel(self.tab)
        self.label_threshold_4.setGeometry(QtCore.QRect(410, 130, 21, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_4.setFont(font)
        self.label_threshold_4.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_4.setObjectName("label_threshold_4")
        self.label_threshold_5 = QtWidgets.QLabel(self.tab)
        self.label_threshold_5.setGeometry(QtCore.QRect(410, 440, 21, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_5.setFont(font)
        self.label_threshold_5.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_5.setObjectName("label_threshold_5")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.button_file_2 = QtWidgets.QPushButton(self.tab_2)
        self.button_file_2.setGeometry(QtCore.QRect(30, 50, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file_2.setFont(font)
        self.button_file_2.setObjectName("button_file_2")
        self.label_threshold_9 = QtWidgets.QLabel(self.tab_2)
        self.label_threshold_9.setGeometry(QtCore.QRect(60, 150, 681, 161))
        font = QtGui.QFont()
        font.setFamily("Kristen ITC")
        font.setPointSize(28)
        self.label_threshold_9.setFont(font)
        self.label_threshold_9.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_9.setObjectName("label_threshold_9")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.button_file_3 = QtWidgets.QPushButton(self.tab_3)
        self.button_file_3.setGeometry(QtCore.QRect(30, 20, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file_3.setFont(font)
        self.button_file_3.setObjectName("button_file_3")
        self.label_brightness = QtWidgets.QLabel(self.tab_3)
        self.label_brightness.setGeometry(QtCore.QRect(220, 140, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_brightness.setFont(font)
        self.label_brightness.setText("")
        self.label_brightness.setObjectName("label_brightness")
        self.label_threshold_7 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_7.setGeometry(QtCore.QRect(40, 110, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_7.setFont(font)
        self.label_threshold_7.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_7.setObjectName("label_threshold_7")
        self.label_img_5 = QtWidgets.QLabel(self.tab_3)
        self.label_img_5.setGeometry(QtCore.QRect(30, 180, 200, 200))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_img_5.setFont(font)
        self.label_img_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_5.setObjectName("label_img_5")
        self.label_img_6 = QtWidgets.QLabel(self.tab_3)
        self.label_img_6.setGeometry(QtCore.QRect(300, 60, 200, 200))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_img_6.setFont(font)
        self.label_img_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_6.setObjectName("label_img_6")
        self.label_threshold_8 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_8.setGeometry(QtCore.QRect(300, 0, 161, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_8.setFont(font)
        self.label_threshold_8.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_8.setObjectName("label_threshold_8")
        self.textEdit_2 = QtWidgets.QTextEdit(self.tab_3)
        self.textEdit_2.setGeometry(QtCore.QRect(520, 310, 201, 101))
        self.textEdit_2.setTabletTracking(False)
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_img_9 = QtWidgets.QLabel(self.tab_3)
        self.label_img_9.setGeometry(QtCore.QRect(290, 310, 200, 200))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_img_9.setFont(font)
        self.label_img_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_9.setObjectName("label_img_9")
        self.label_threshold_12 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_12.setGeometry(QtCore.QRect(290, 260, 161, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_12.setFont(font)
        self.label_threshold_12.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_12.setObjectName("label_threshold_12")
        self.tabWidget.addTab(self.tab_3, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 801, 21))
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
        self.button_original.setText(_translate("MainWindow", "Original"))
        self.label_img_1.setText(_translate("MainWindow", "img"))
        self.button_trapezoidal.setText(_translate("MainWindow", "Trapezoidal"))
        self.button_wavy.setText(_translate("MainWindow", "Wavy"))
        self.button_circular.setText(_translate("MainWindow", "Circular"))
        self.button_file_1.setText(_translate("MainWindow", "choose img"))
        self.label_img_2.setText(_translate("MainWindow", "img"))
        self.label_img_7.setText(_translate("MainWindow", "img"))
        self.label_img_8.setText(_translate("MainWindow", "img"))
        self.label_threshold_4.setText(_translate("MainWindow", "X"))
        self.label_threshold_5.setText(_translate("MainWindow", "X"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Part1"))
        self.button_file_2.setText(_translate("MainWindow", "choose img"))
        self.label_threshold_9.setText(_translate("MainWindow", "I haven\'t finished this part yet. Sorry."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Part2"))
        self.button_file_3.setText(_translate("MainWindow", "choose img"))
        self.label_threshold_7.setText(_translate("MainWindow", "Original"))
        self.label_img_5.setText(_translate("MainWindow", "img"))
        self.label_img_6.setText(_translate("MainWindow", "img"))
        self.label_threshold_8.setText(_translate("MainWindow", "Edge"))
        self.label_img_9.setText(_translate("MainWindow", "img"))
        self.label_threshold_12.setText(_translate("MainWindow", "Hough transform"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Part3"))
