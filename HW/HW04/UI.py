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
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 781, 551))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tabWidget.setElideMode(QtCore.Qt.ElideLeft)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.button_file = QtWidgets.QPushButton(self.tab)
        self.button_file.setGeometry(QtCore.QRect(20, 20, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file.setFont(font)
        self.button_file.setObjectName("button_file")
        self.button_fft = QtWidgets.QPushButton(self.tab)
        self.button_fft.setGeometry(QtCore.QRect(180, 20, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_fft.setFont(font)
        self.button_fft.setObjectName("button_fft")
        self.label_img_1 = QtWidgets.QLabel(self.tab)
        self.label_img_1.setGeometry(QtCore.QRect(30, 130, 200, 150))
        self.label_img_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_1.setObjectName("label_img_1")
        self.label_img_2 = QtWidgets.QLabel(self.tab)
        self.label_img_2.setGeometry(QtCore.QRect(410, 20, 200, 150))
        self.label_img_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_2.setObjectName("label_img_2")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_img_4 = QtWidgets.QLabel(self.tab_2)
        self.label_img_4.setGeometry(QtCore.QRect(280, 130, 200, 150))
        self.label_img_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_4.setObjectName("label_img_4")
        self.label_threshold_2 = QtWidgets.QLabel(self.tab_2)
        self.label_threshold_2.setGeometry(QtCore.QRect(40, 60, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        self.label_threshold_2.setFont(font)
        self.label_threshold_2.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_2.setObjectName("label_threshold_2")
        self.button_filter_1 = QtWidgets.QPushButton(self.tab_2)
        self.button_filter_1.setGeometry(QtCore.QRect(590, 100, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_filter_1.setFont(font)
        self.button_filter_1.setObjectName("button_filter_1")
        self.button_filter_2 = QtWidgets.QPushButton(self.tab_2)
        self.button_filter_2.setGeometry(QtCore.QRect(590, 180, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_filter_2.setFont(font)
        self.button_filter_2.setObjectName("button_filter_2")
        self.button_filter_3 = QtWidgets.QPushButton(self.tab_2)
        self.button_filter_3.setGeometry(QtCore.QRect(590, 250, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_filter_3.setFont(font)
        self.button_filter_3.setObjectName("button_filter_3")
        self.label_img_3 = QtWidgets.QLabel(self.tab_2)
        self.label_img_3.setGeometry(QtCore.QRect(40, 130, 200, 150))
        self.label_img_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_3.setObjectName("label_img_3")
        self.label_threshold_3 = QtWidgets.QLabel(self.tab_2)
        self.label_threshold_3.setGeometry(QtCore.QRect(290, 60, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        self.label_threshold_3.setFont(font)
        self.label_threshold_3.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_3.setObjectName("label_threshold_3")
        self.button_file_2 = QtWidgets.QPushButton(self.tab_2)
        self.button_file_2.setGeometry(QtCore.QRect(30, 10, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file_2.setFont(font)
        self.button_file_2.setObjectName("button_file_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.button_file_3 = QtWidgets.QPushButton(self.tab_3)
        self.button_file_3.setGeometry(QtCore.QRect(20, 20, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file_3.setFont(font)
        self.button_file_3.setObjectName("button_file_3")
        self.label_threshold_4 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_4.setGeometry(QtCore.QRect(20, 90, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_4.setFont(font)
        self.label_threshold_4.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_4.setObjectName("label_threshold_4")
        self.label_brightness = QtWidgets.QLabel(self.tab_3)
        self.label_brightness.setGeometry(QtCore.QRect(200, 150, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_brightness.setFont(font)
        self.label_brightness.setText("")
        self.label_brightness.setObjectName("label_brightness")
        self.horizontalSlider_brightness = QtWidgets.QSlider(self.tab_3)
        self.horizontalSlider_brightness.setGeometry(QtCore.QRect(20, 150, 160, 22))
        self.horizontalSlider_brightness.setMinimum(-255)
        self.horizontalSlider_brightness.setMaximum(255)
        self.horizontalSlider_brightness.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_brightness.setObjectName("horizontalSlider_brightness")
        self.label_brightness_2 = QtWidgets.QLabel(self.tab_3)
        self.label_brightness_2.setGeometry(QtCore.QRect(200, 150, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_brightness_2.setFont(font)
        self.label_brightness_2.setObjectName("label_brightness_2")
        self.label_brightness_3 = QtWidgets.QLabel(self.tab_3)
        self.label_brightness_3.setGeometry(QtCore.QRect(200, 260, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_brightness_3.setFont(font)
        self.label_brightness_3.setObjectName("label_brightness_3")
        self.horizontalSlider_brightness_2 = QtWidgets.QSlider(self.tab_3)
        self.horizontalSlider_brightness_2.setGeometry(QtCore.QRect(20, 260, 160, 22))
        self.horizontalSlider_brightness_2.setMinimum(-255)
        self.horizontalSlider_brightness_2.setMaximum(255)
        self.horizontalSlider_brightness_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_brightness_2.setObjectName("horizontalSlider_brightness_2")
        self.label_threshold_5 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_5.setGeometry(QtCore.QRect(20, 200, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_5.setFont(font)
        self.label_threshold_5.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_5.setObjectName("label_threshold_5")
        self.label_brightness_4 = QtWidgets.QLabel(self.tab_3)
        self.label_brightness_4.setGeometry(QtCore.QRect(200, 380, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_brightness_4.setFont(font)
        self.label_brightness_4.setObjectName("label_brightness_4")
        self.label_threshold_6 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_6.setGeometry(QtCore.QRect(20, 320, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_6.setFont(font)
        self.label_threshold_6.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_6.setObjectName("label_threshold_6")
        self.horizontalSlider_brightness_3 = QtWidgets.QSlider(self.tab_3)
        self.horizontalSlider_brightness_3.setGeometry(QtCore.QRect(20, 380, 160, 22))
        self.horizontalSlider_brightness_3.setMinimum(-255)
        self.horizontalSlider_brightness_3.setMaximum(255)
        self.horizontalSlider_brightness_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_brightness_3.setObjectName("horizontalSlider_brightness_3")
        self.label_threshold_7 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_7.setGeometry(QtCore.QRect(360, 30, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_7.setFont(font)
        self.label_threshold_7.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_7.setObjectName("label_threshold_7")
        self.label_img_5 = QtWidgets.QLabel(self.tab_3)
        self.label_img_5.setGeometry(QtCore.QRect(360, 80, 200, 150))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_img_5.setFont(font)
        self.label_img_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_5.setObjectName("label_img_5")
        self.label_img_6 = QtWidgets.QLabel(self.tab_3)
        self.label_img_6.setGeometry(QtCore.QRect(360, 310, 200, 150))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_img_6.setFont(font)
        self.label_img_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_6.setObjectName("label_img_6")
        self.label_threshold_8 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_8.setGeometry(QtCore.QRect(360, 260, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_8.setFont(font)
        self.label_threshold_8.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_8.setObjectName("label_threshold_8")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.tabWidget.addTab(self.tab_4, "")
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
        self.button_fft.setText(_translate("MainWindow", "FFT"))
        self.label_img_1.setText(_translate("MainWindow", "img"))
        self.label_img_2.setText(_translate("MainWindow", "result"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Part1"))
        self.label_img_4.setText(_translate("MainWindow", "result"))
        self.label_threshold_2.setText(_translate("MainWindow", "Origin"))
        self.button_filter_1.setText(_translate("MainWindow", "  Ideal  filter"))
        self.button_filter_2.setText(_translate("MainWindow", "Butterworth filter"))
        self.button_filter_3.setText(_translate("MainWindow", "Gaussian filter"))
        self.label_img_3.setText(_translate("MainWindow", "img"))
        self.label_threshold_3.setText(_translate("MainWindow", "Origin"))
        self.button_file_2.setText(_translate("MainWindow", "choose img"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Part2"))
        self.button_file_3.setText(_translate("MainWindow", "choose img"))
        self.label_threshold_4.setText(_translate("MainWindow", "Gamma_H"))
        self.label_brightness_2.setText(_translate("MainWindow", "value"))
        self.label_brightness_3.setText(_translate("MainWindow", "value"))
        self.label_threshold_5.setText(_translate("MainWindow", "Gamma_D"))
        self.label_brightness_4.setText(_translate("MainWindow", "value"))
        self.label_threshold_6.setText(_translate("MainWindow", "D0"))
        self.label_threshold_7.setText(_translate("MainWindow", "Origin"))
        self.label_img_5.setText(_translate("MainWindow", "img"))
        self.label_img_6.setText(_translate("MainWindow", "img"))
        self.label_threshold_8.setText(_translate("MainWindow", "homomorphic"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Part3"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Part4"))