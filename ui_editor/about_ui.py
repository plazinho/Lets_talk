# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/about.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDesktopWidget


# Окно, которое открывается по кнопке "About"
class Ui_About(object):
    def setupUi(self, About):
        About.setObjectName("About")
        About.resize(1031, 861)
        About.move(QDesktopWidget().availableGeometry().center() + QtCore.QPoint(+250-1031, -430))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(About.sizePolicy().hasHeightForWidth())
        About.setSizePolicy(sizePolicy)
        About.setMinimumSize(QtCore.QSize(1031, 861))
        About.setMaximumSize(QtCore.QSize(1031, 861))
        self.label_3 = QtWidgets.QLabel(About)
        self.label_3.setGeometry(QtCore.QRect(24, 0, 991, 51))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setMinimumSize(QtCore.QSize(991, 51))
        self.label_3.setMaximumSize(QtCore.QSize(991, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_3.setFont(font)
        self.label_3.setToolTipDuration(-1)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.scrollArea_2 = QtWidgets.QScrollArea(About)
        self.scrollArea_2.setGeometry(QtCore.QRect(40, 60, 971, 791))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea_2.sizePolicy().hasHeightForWidth())
        self.scrollArea_2.setSizePolicy(sizePolicy)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, -1640, 948, 2496))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_5 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setMinimumSize(QtCore.QSize(901, 108))
        self.label_5.setMaximumSize(QtCore.QSize(901, 108))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setWordWrap(True)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_3.addWidget(self.label_5)
        self.label_4 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setMinimumSize(QtCore.QSize(761, 411))
        self.label_4.setMaximumSize(QtCore.QSize(761, 411))
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("pictures/Basic_about_cut-painted.jpg"))
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.label_6 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setMinimumSize(QtCore.QSize(901, 244))
        self.label_6.setMaximumSize(QtCore.QSize(901, 244))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_3.addWidget(self.label_6)
        self.label_9 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setMinimumSize(QtCore.QSize(391, 28))
        self.label_9.setMaximumSize(QtCore.QSize(391, 28))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_9.setFont(font)
        self.label_9.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_3.addWidget(self.label_9)
        self.label_7 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setMinimumSize(QtCore.QSize(761, 411))
        self.label_7.setMaximumSize(QtCore.QSize(761, 411))
        self.label_7.setText("")
        self.label_7.setPixmap(QtGui.QPixmap("pictures/model_start-painted.jpg"))
        self.label_7.setScaledContents(True)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_3.addWidget(self.label_7)
        self.label_8 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setMinimumSize(QtCore.QSize(766, 21))
        self.label_8.setMaximumSize(QtCore.QSize(766, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setWordWrap(True)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_3.addWidget(self.label_8)
        self.label_10 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setMinimumSize(QtCore.QSize(761, 411))
        self.label_10.setMaximumSize(QtCore.QSize(761, 411))
        self.label_10.setText("")
        self.label_10.setPixmap(QtGui.QPixmap("pictures/model_stop-painted.jpg"))
        self.label_10.setScaledContents(True)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_3.addWidget(self.label_10)
        self.label_11 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setMinimumSize(QtCore.QSize(617, 21))
        self.label_11.setMaximumSize(QtCore.QSize(617, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setWordWrap(True)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_3.addWidget(self.label_11)
        self.label_12 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setMinimumSize(QtCore.QSize(761, 411))
        self.label_12.setMaximumSize(QtCore.QSize(761, 411))
        self.label_12.setText("")
        self.label_12.setPixmap(QtGui.QPixmap("pictures/noneNan.jpg"))
        self.label_12.setScaledContents(True)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_3.addWidget(self.label_12)
        self.label_13 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_13.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy)
        self.label_13.setMinimumSize(QtCore.QSize(924, 343))
        self.label_13.setMaximumSize(QtCore.QSize(924, 343))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setMouseTracking(False)
        self.label_13.setWordWrap(True)
        self.label_13.setObjectName("label_13")
        self.verticalLayout_3.addWidget(self.label_13)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.retranslateUi(About)
        QtCore.QMetaObject.connectSlotsByName(About)

    def retranslateUi(self, About):
        _translate = QtCore.QCoreApplication.translate
        About.setWindowTitle(_translate("About", "Let\'s Talk - About"))
        self.label_3.setText(_translate("About", "Инструкция пользователя"))
        self.label_5.setText(_translate("About", "<html><head/><body><p>Программа<span style=\" font-weight:600;\"> Let\'s talk </span>создана для распознавания языка жестов ASL (American Sign Language). </p><p>На данной странице представлено рабочее пространство программы Let\'s talk.</p><p>Основное окно программы отображает изображение, получаемое с камеры устройства, на котором используется программа. Справа от него расположено окно редактирования и записи.</p></body></html>"))
        self.label_6.setText(_translate("About", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Основное окно</span></p><p>В верхней части Основного окна расположена &quot;Строка предсказаний&quot;, отображающая знаки, распознанные программой (но не записанные). Под ней, в левом верхнем углу, расположены текущие предсказания, отсортированные в порядке убывания вероятности (по данным модели).</p><p>В правом верхнем углу расположен индикатор - Режим работы, отображающий текущий режим работы программы. </p><p>Красный цвет индикатора - программа в режиме Model_start - распознанные знаки сохраняются и записываются (установлен по умолчанию).</p><p>Зеленый цвет индикатора - программа в режиме Model_stop, в данном режиме модель не записывает текущие предсказания, ранее распознанные знаки сохраняются.</p></body></html>"))
        self.label_9.setText(_translate("About", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Жесты управления программой:</span></p></body></html>"))
        self.label_8.setText(_translate("About", "<html><head/><body><p><span style=\" font-size:10pt;\">model_start - переводит программу в режим распознования и записи (красный цвет индикатора)</span></p></body></html>"))
        self.label_11.setText(_translate("About", "<html><head/><body><p><span style=\" font-size:10pt;\">model_stop - переводит программу в режим паузы (зеленый цвет индикатора)</span></p></body></html>"))
        self.label_13.setText(_translate("About", "<html><head/><body><p>NaN - знак, который позволяет записать распознанные знаки (сохранить предложение), отображаемые в &quot;Окне предсказаний&quot;, записанные знаки отображаются в отдельном окне - Редактирования и записи.</p><p>Со списком доступных знаков можно ознакомиться, перейдя на вкладку &quot;Available signs&quot; в главном меню. </p><p><span style=\" font-size:14pt; font-weight:600;\">Окно Редактирования и записи</span></p><p>Данное окно запускается одновременно с Основным окном.</p><p>В верхней части окна расположены кнопки:</p><p>Duplicate last sign - позволяет повторить последний распознанный, но не записанный знак. Данная функция была реализована в связи с программным ограничением ввода повторяющися знаков для обеспечения корректной работы программы.</p><p>Delete_last_sign - позволяет удалить последний введенный знак. Обе кнопки работают только в режиме Model_stop.</p><p>Под кнопками расположено окно вывода записанного текста, текст доступен для копирования, но недоступен для редактирования.</p></body></html>"))
