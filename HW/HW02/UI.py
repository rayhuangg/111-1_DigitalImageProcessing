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
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.button_file = QtWidgets.QPushButton(self.centralwidget)
        self.button_file.setGeometry(QtCore.QRect(70, 50, 111, 31))
        self.button_file.setObjectName("button_file")
        self.show_file_path1 = QtWidgets.QTextEdit(self.centralwidget)
        self.show_file_path1.setGeometry(QtCore.QRect(240, 40, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.show_file_path1.setFont(font)
        self.show_file_path1.setObjectName("show_file_path1")
        self.label_img = QtWidgets.QLabel(self.centralwidget)
        self.label_img.setGeometry(QtCore.QRect(80, 110, 200, 150))
        self.label_img.setObjectName("label_img")
        self.label_hist = QtWidgets.QLabel(self.centralwidget)
        self.label_hist.setGeometry(QtCore.QRect(310, 110, 200, 150))
        self.label_hist.setObjectName("label_hist")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 270, 531, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(220, 340, 141, 31))
        self.textBrowser.setObjectName("textBrowser")
        self.label_hist_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_2.setGeometry(QtCore.QRect(310, 400, 200, 150))
        self.label_hist_2.setObjectName("label_hist_2")
        self.label_img_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_2.setGeometry(QtCore.QRect(80, 400, 200, 150))
        self.label_img_2.setObjectName("label_img_2")
        self.label_hist_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_3.setGeometry(QtCore.QRect(310, 620, 200, 150))
        self.label_hist_3.setObjectName("label_hist_3")
        self.label_img_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_3.setGeometry(QtCore.QRect(80, 620, 200, 150))
        self.label_img_3.setObjectName("label_img_3")
        self.label_hist_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_4.setGeometry(QtCore.QRect(310, 830, 200, 150))
        self.label_hist_4.setObjectName("label_hist_4")
        self.label_img_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_4.setGeometry(QtCore.QRect(80, 830, 200, 150))
        self.label_img_4.setObjectName("label_img_4")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(220, 570, 231, 31))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_3.setGeometry(QtCore.QRect(220, 780, 141, 31))
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(540, 0, 31, 1011))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.textBrowser_4 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_4.setGeometry(QtCore.QRect(70, 10, 61, 31))
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.label_img_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_5.setGeometry(QtCore.QRect(680, 130, 200, 150))
        self.label_img_5.setObjectName("label_img_5")
        self.button_2_2_grayscale = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_2_grayscale.setGeometry(QtCore.QRect(70, 290, 111, 41))
        self.button_2_2_grayscale.setObjectName("button_2_2_grayscale")
        self.button_2_3_histogram = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_3_histogram.setGeometry(QtCore.QRect(70, 340, 111, 41))
        self.button_2_3_histogram.setObjectName("button_2_3_histogram")
        self.button_2_4_threshold = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_4_threshold.setGeometry(QtCore.QRect(630, 40, 111, 41))
        self.button_2_4_threshold.setObjectName("button_2_4_threshold")
        self.button_2_5_enlarge = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_5_enlarge.setGeometry(QtCore.QRect(640, 520, 111, 41))
        self.button_2_5_enlarge.setObjectName("button_2_5_enlarge")
        self.label_img_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_6.setGeometry(QtCore.QRect(680, 610, 200, 150))
        self.label_img_6.setObjectName("label_img_6")
        self.button_2_5_shrink = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_5_shrink.setGeometry(QtCore.QRect(800, 520, 111, 41))
        self.button_2_5_shrink.setObjectName("button_2_5_shrink")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(570, 400, 491, 20))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.button_2_6_brightness = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_6_brightness.setGeometry(QtCore.QRect(1130, 30, 201, 41))
        self.button_2_6_brightness.setObjectName("button_2_6_brightness")
        self.label_img_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_7.setGeometry(QtCore.QRect(1140, 180, 200, 150))
        self.label_img_7.setObjectName("label_img_7")
        self.label_hist_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_7.setGeometry(QtCore.QRect(1140, 340, 200, 150))
        self.label_hist_7.setObjectName("label_hist_7")
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_5.setGeometry(QtCore.QRect(1170, 90, 101, 41))
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.textBrowser_6 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_6.setGeometry(QtCore.QRect(1460, 90, 101, 41))
        self.textBrowser_6.setObjectName("textBrowser_6")
        self.label_hist_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_8.setGeometry(QtCore.QRect(1430, 340, 200, 150))
        self.label_hist_8.setObjectName("label_hist_8")
        self.label_img_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_8.setGeometry(QtCore.QRect(1430, 180, 200, 150))
        self.label_img_8.setObjectName("label_img_8")
        self.horizontalSlider_threshold = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_threshold.setGeometry(QtCore.QRect(770, 60, 160, 22))
        self.horizontalSlider_threshold.setMaximum(255)
        self.horizontalSlider_threshold.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_threshold.setObjectName("horizontalSlider_threshold")
        self.horizontalSlider_brightness = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_brightness.setGeometry(QtCore.QRect(1140, 140, 160, 22))
        self.horizontalSlider_brightness.setMinimum(-255)
        self.horizontalSlider_brightness.setMaximum(255)
        self.horizontalSlider_brightness.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_brightness.setObjectName("horizontalSlider_brightness")
        self.horizontalSlider_contrast = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_contrast.setGeometry(QtCore.QRect(1430, 140, 160, 22))
        self.horizontalSlider_contrast.setMinimum(-100)
        self.horizontalSlider_contrast.setMaximum(100)
        self.horizontalSlider_contrast.setProperty("value", 0)
        self.horizontalSlider_contrast.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_contrast.setObjectName("horizontalSlider_contrast")
        self.label_threshold = QtWidgets.QLabel(self.centralwidget)
        self.label_threshold.setGeometry(QtCore.QRect(820, 40, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_threshold.setFont(font)
        self.label_threshold.setObjectName("label_threshold")
        self.label_brightness = QtWidgets.QLabel(self.centralwidget)
        self.label_brightness.setGeometry(QtCore.QRect(1320, 140, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_brightness.setFont(font)
        self.label_brightness.setObjectName("label_brightness")
        self.label_contrast = QtWidgets.QLabel(self.centralwidget)
        self.label_contrast.setGeometry(QtCore.QRect(1600, 140, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_contrast.setFont(font)
        self.label_contrast.setObjectName("label_contrast")
        self.button_2_6_contrast = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_6_contrast.setGeometry(QtCore.QRect(1410, 30, 201, 41))
        self.button_2_6_contrast.setObjectName("button_2_6_contrast")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(1060, 0, 31, 1041))
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(1070, 510, 701, 20))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.button_2_7_histogram_equalazation = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_7_histogram_equalazation.setGeometry(QtCore.QRect(1110, 540, 201, 41))
        self.button_2_7_histogram_equalazation.setObjectName("button_2_7_histogram_equalazation")
        self.label_threshold_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_threshold_2.setGeometry(QtCore.QRect(1180, 610, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_threshold_2.setFont(font)
        self.label_threshold_2.setObjectName("label_threshold_2")
        self.label_threshold_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_threshold_3.setGeometry(QtCore.QRect(1470, 610, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_threshold_3.setFont(font)
        self.label_threshold_3.setObjectName("label_threshold_3")
        self.label_img_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_10.setGeometry(QtCore.QRect(1410, 660, 200, 150))
        self.label_img_10.setObjectName("label_img_10")
        self.label_img_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_9.setGeometry(QtCore.QRect(1120, 660, 200, 150))
        self.label_img_9.setObjectName("label_img_9")
        self.label_hist_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_9.setGeometry(QtCore.QRect(1120, 820, 200, 150))
        self.label_hist_9.setObjectName("label_hist_9")
        self.label_hist_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_10.setGeometry(QtCore.QRect(1410, 820, 200, 150))
        self.label_hist_10.setObjectName("label_hist_10")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_file.setText(_translate("MainWindow", "2-1 choose img"))
        self.show_file_path1.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:11pt;\"><br /></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:11pt;\"><br /></p></body></html>"))
        self.label_img.setText(_translate("MainWindow", "img"))
        self.label_hist.setText(_translate("MainWindow", "histogram"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">GRAY = (R+G+B)/3.0</p></body></html>"))
        self.label_hist_2.setText(_translate("MainWindow", "histogram"))
        self.label_img_2.setText(_translate("MainWindow", "img"))
        self.label_hist_3.setText(_translate("MainWindow", "histogram"))
        self.label_img_3.setText(_translate("MainWindow", "img"))
        self.label_hist_4.setText(_translate("MainWindow", "histogram"))
        self.label_img_4.setText(_translate("MainWindow", "img"))
        self.textBrowser_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">GRAY = 0.299*R + 0.587*G + 0.114*B</p></body></html>"))
        self.textBrowser_3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">image subtraction</p></body></html>"))
        self.textBrowser_4.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2-1</p></body></html>"))
        self.label_img_5.setText(_translate("MainWindow", "img"))
        self.button_2_2_grayscale.setText(_translate("MainWindow", "2-2 grayscale"))
        self.button_2_3_histogram.setText(_translate("MainWindow", "2-3 histogram"))
        self.button_2_4_threshold.setText(_translate("MainWindow", "2-4 threshold"))
        self.button_2_5_enlarge.setText(_translate("MainWindow", "2-5 enlarge"))
        self.label_img_6.setText(_translate("MainWindow", "img"))
        self.button_2_5_shrink.setText(_translate("MainWindow", "2-5 shrink"))
        self.button_2_6_brightness.setText(_translate("MainWindow", "2-6 brightness"))
        self.label_img_7.setText(_translate("MainWindow", "img"))
        self.label_hist_7.setText(_translate("MainWindow", "histogram"))
        self.textBrowser_5.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">brightness</p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">(-255~255)</p></body></html>"))
        self.textBrowser_6.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">contrast</p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">(-100~100)</p></body></html>"))
        self.label_hist_8.setText(_translate("MainWindow", "histogram"))
        self.label_img_8.setText(_translate("MainWindow", "img"))
        self.label_threshold.setText(_translate("MainWindow", "value"))
        self.label_brightness.setText(_translate("MainWindow", "value"))
        self.label_contrast.setText(_translate("MainWindow", "value"))
        self.button_2_6_contrast.setText(_translate("MainWindow", "2-6 constrast"))
        self.button_2_7_histogram_equalazation.setText(_translate("MainWindow", "2-7 histogram equalization"))
        self.label_threshold_2.setText(_translate("MainWindow", "Before"))
        self.label_threshold_3.setText(_translate("MainWindow", "After"))
        self.label_img_10.setText(_translate("MainWindow", "img"))
        self.label_img_9.setText(_translate("MainWindow", "img"))
        self.label_hist_9.setText(_translate("MainWindow", "histogram"))
        self.label_hist_10.setText(_translate("MainWindow", "histogram"))
