# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:18:16 2020

@author: Kaking
"""


import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt

class Canvas(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        
        self.label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(400, 300)
        pixmap.fill(QtGui.QColor('white'))
        self.setPixmap(pixmap)
        
        self.last_x, self.last_y = None, None
        self.pen_colour = QtGui.QColor('#000000')

    
    def set_pen_colours(self, c):
        self.pen_colour = QtGui.QColor(c)
        
    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return
        
        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(4)
        p.setColor(self.pen_colour)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()
        
        self.last_x = e.x()
        self.last_y = e.y()
        
    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None
        

COLOURS =['#000000', '#ffffff']

class QPaletteButton(QtWidgets.QPushButton):
    def __init__(self, colour):
        super().__init__()
        self.setFixedSize(QtCore.QSize(24, 24))
        self.colour = colour
        self.setStyleSheet("background-color: %s" % colour)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.canvas = Canvas()
        
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout()
        w.setLayout(l)
        l.addWidget(self.canvas)
        
        palette = QtWidgets.QHBoxLayout()
        self.add_palette_buttons(palette)
        l.addLayout(palette)
        
        self.setCentralWidget(w)
        
    def add_palette_buttons(self, layout):
        for c in COLOURS:
            b = QPaletteButton(c)
            b.pressed.connect(lambda c=c: self.canvas.set_pen_colours(c))
            layout.addWidget(b)
            
     
        
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
        