# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:11:35 2020

@author: Kaking
"""

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMenuBar, QMenu, QAction, QFileDialog, QLabel, QMessageBox
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
from PIL import Image
from StyleTransferTensor import *
from Mask_Transfer import *
from SemanticSegmentation import *
import matplotlib.pyplot as plt
from imageio import imread, imsave
import sys


# Paint Layer
class Layer(QtWidgets.QGraphicsRectItem):
    
    # The 2 States during the Painting function
    DrawState, EraseState = range(2)
    
    # Inialising the main components of the painting program
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_state = Layer.DrawState
        self.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        
        self.line_eraser = QtCore.QLineF()
        self.line_draw = QtCore.QLineF()
        self.pixmap = QtGui.QPixmap()
        self.pen_colour = QtGui.QColor('black')
        
    # Resets the painting layer when new image has been opened 
    def reset(self):
        r = self.parentItem().pixmap().rect()
        self.setRect(QtCore.QRectF(r))
        self.pixmap = QtGui.QPixmap(r.size())
        self.pixmap.fill(QtCore.Qt.transparent)
        
    # Creates a pixmap to draw lines
    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.save()
        painter.drawPixmap(QtCore.QPoint(), self.pixmap)
        painter.restore()
        
    # Mouse events to draw lines on the drawing pixmap on mouse click
    def mousePressEvent(self, event):
        if self.current_state == Layer.EraseState:
            self.clear(event.pos().toPoint())
        elif self.current_state == Layer.DrawState:
             self.line_draw.setP1(event.pos())
             self.line_draw.setP2(event.pos())
        super().mousePressEvent(event)
        event.accept()
    
    # Allows the lines/eraser to move with the mouse as it moves
    def mouseMoveEvent(self, event):
        if self.current_state == Layer.EraseState:
            self.clear(event.pos().toPoint())
        elif self.current_state == Layer.DrawState:
            self.line_draw.setP2(event.pos())
            self.draw_line(self.line_draw, QtGui.QPen(self.pen_colour, self.pen_thickness))
            self.line_draw.setP1(event.pos())
        super().mouseMoveEvent(event)
    
    # Drawing function to create lines
    def draw_line(self, line, pen):
        painter = QtGui.QPainter(self.pixmap)
        painter.setPen(pen)
        painter.drawLine(line)
        painter.end()
        self.update()
    
    # Clears the drawing layer
    def clear(self, pos):
        painter = QtGui.QPainter(self.pixmap)
        r = QtCore.QRect(QtCore.QPoint(), 10 * QtCore.QSize())
        r.moveCenter(pos)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        painter.eraseRect(r)
        painter.end()
        self.update()
        
    # Inializes the pen thickness
    def pen_thickness(self):
        return self.pen_thickness
    
    # Assigns the thickness based on the toolbar selection
    def pen_thickness(self, thickness):
        self.pen_thickness = thickness
        
    # Inializes the pen colour    
    def pen_colour(self):
        return self.pen_colour
    
    # Assigns the colour, in this case it will be black
    def pen_colour(self, colour):
        self.pen_colour = colour
        
    # Inializes state changes     
    def current_state(self):
        return self.current_state
    
    # Allows changes in states by switching back and forth between drawing and eraser mode
    def current_state(self, state):
        self.current_state = state

# Graphic Box that allows two layers - Image layer and Drawing layer
class GraphicsView(QtWidgets.QGraphicsView):
    
    # Inializing the main components such as background layer and foreground layer and removing the scroll bars
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        self.background_item = QtWidgets.QGraphicsPixmapItem()
        self.foreground_item = Layer(self.background_item)
        
        self.scene().addItem(self.background_item)
        
    # Sets the image to the background layer and resets the drawing layer
    # The imagebox is resized to match the image size of the stylized product
    def set_image(self, image, width, height):
        self.scene().setSceneRect(0, 0, width, height)
        self.background_item.setPixmap(image)
        self.foreground_item.reset()
        self.fitInView(self.background_item)
        self.centerOn(self.background_item)

# The Main program that contains all the components
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        
       # Thickness Pen Slider
        self.pen_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal,
                                            minimum=3, maximum=30,
                                            value=5,
                                            focusPolicy=QtCore.Qt.StrongFocus,
                                            tickPosition=QtWidgets.QSlider.TicksBothSides,
                                            tickInterval=1,
                                            singleStep=1,
                                            valueChanged=self.changedThickness,)
        
        # Eraser checkbox that triggers the eraser state change
        self.eraser_checkbox = QtWidgets.QCheckBox(self.tr("Eraser"), stateChanged=self.stateChanged)
        
        # Sets the graphic box
        self.view = GraphicsView()
        self.view.foreground_item.pen_thickness = self.pen_slider.value()
        
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(w)
        l.addWidget(self.view)
        
        toollist = QtWidgets.QHBoxLayout()
        l.addLayout(toollist)
        
        # Eraser and the pen slider
        toollist.addWidget(self.pen_slider)
        toollist.addWidget(self.eraser_checkbox)
        
        # Auto Transfer Button - Vanilla Style Transfer
        styleButton = QtWidgets.QPushButton()
        styleButton.setText('Style Transfer')
        styleButton.move(20, 50)
        styleButton.setFixedSize(QtCore.QSize(80, 30))
        styleButton.clicked.connect(self.style_transfer)
        toollist.addWidget(styleButton)
        
        # Manual Transfer Button triggers Manual Mask creation with User drawings
        manualButton = QtWidgets.QPushButton()
        manualButton.setText('Manual Mask')
        manualButton.move(20, 50)
        manualButton.setFixedSize(QtCore.QSize(80, 30))
        manualButton.clicked.connect(self.openFile)
        toollist.addWidget(manualButton)
        
        # Auto Transfer Button that triggers Segmentation with the Mask Transfer
        autoMaskButton = QtWidgets.QPushButton()
        autoMaskButton.setText('Auto Mask')
        autoMaskButton.move(40, 50)
        autoMaskButton.setFixedSize(QtCore.QSize(80, 30))
        autoMaskButton.clicked.connect(self.autoMask)
        toollist.addWidget(autoMaskButton)
        
        
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")
        
        saveAction = QAction(QIcon("save.png"), "Save", self)
        saveAction.setShortcut("Ctrl+S")
        fileMenu.addAction(saveAction)
        saveAction.triggered.connect(self.save)
        
        self.setCentralWidget(w)
        
        self.resize(600,600)
        
    # Assigns new thickness 
    def changedThickness(self, value):
        self.view.foreground_item.pen_thickness = value
    
    def stateChanged(self, state):
        self.view.foreground_item.current_state = (Layer.EraseState if state == QtCore.Qt.Checked else Layer.DrawState)
    
    # Saves the user drawn mask and performs mask transfer on the 
    # newly created mask and displays the product
    def save(self):
        
        # Puts a blank image on the background to erase the background image
        # This allows Binarization of the mask image.
        white = Image.open('blank.jpg')
        white = white.convert("RGBA")
        data = white.tobytes("raw", "RGBA")
        qim = QtGui.QImage(data, 512, 512, QtGui.QImage.Format_ARGB32)
        white_pix = QtGui.QPixmap.fromImage(qim)
        self.view.background_item.setPixmap(white_pix)
        
        # Renders the graphic view to an image which allows it to be saved.
        rect = self.view.rect()
        image = QImage(rect.size(), QImage.Format_ARGB32_Premultiplied)
        painter = QPainter(image)
        
        self.view.render(painter)
        
        # Saves file in a place of their choice using file explorer
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPG(*.jpg *.jpg);;All Files(*.*) ")
        image.save(filePath)
        
        msg = QMessageBox()
        msg.setText("Re-open the content image")
        msg.exec_()
        
        # Reloads the content image in preparation of mask transfer
        contentName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        
        # Runs the mask transfer with the mask image, content image and the vanilla style transfer product
        img = run_mask_transfer('Products/100-stylephoto.jpg', contentName, filePath)
        product = imread('product.jpg')
        plt.imshow(product)
        
    # Opens the image file of the users choice and performs Style Transfer
    # This is to ensure that the mask image and the product image is the same size
    def openFile(self):
        
        msg = QMessageBox()
        msg.setText("Choose Content Image")
        msg.exec_()
        
        # Opens image file of their choice using file explorer
        contentName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        
        imageCheck = Image.open(contentName)
        
        # Content image - Width and Height
        w, h = imageCheck.size
        
        msg = QMessageBox()
        msg.setText("Choose Style Image")
        msg.exec_()
        styleName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        
        # Performs Vanilla Style Transfer to create a product image and stores the width and height of the product image
        # The width and height is used to compare the product image and content image sizes.
        width, height = run_style_transfer(contentName, styleName)
        
            
        # If the width/height of the content image is bigger than the product image,
        # it is resized to match the product image.
        if w > width or h > height:
            resized = imageCheck.resize((width, height))
            resized.save("resized.jpg")
            pixmap = QtGui.QPixmap("resized.jpg")
            self.view.set_image(pixmap, width, height)
            
            msg = QMessageBox()
            msg.setText("Once region is finalized, File and save the mask.")
            msg.exec_()
        
        else:
            
            pixmap = QtGui.QPixmap(contentName)
            pixmap.scaled(width, height, QtCore.Qt.IgnoreAspectRatio)
            self.view.set_image(pixmap, width, height)
            
            print(width, height)
            
            msg = QMessageBox()
            msg.setText("Once region is finalized, File and save the mask.")
            msg.exec_()
         
        
    # Function to perform the Vanilla Style Transfer
    def style_transfer(self):
        
        msg = QMessageBox()
        msg.setText("Choose Content Image")
        msg.exec_()
        contentName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        
        msg = QMessageBox()
        msg.setText("Choose Style Image")
        msg.exec_()
        styleName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        
        run_style_transfer(contentName, styleName)
        img = imread('Products/100-stylephoto.jpg')
        plt.imshow(img)
        
    # Function to perform Mask Transfer with Semantic Segmentation  
    def autoMask(self):
        
        msg = QMessageBox()
        msg.setText("Choose Content Image")
        msg.exec_()
        contentName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        
        
        msg = QMessageBox()
        msg.setText("Choose Style Image")
        msg.exec_()
        styleName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        
        width, height = run_style_transfer(contentName, styleName)
        
        # Performs semantic segmentation to get the region of the important region in the image
        # The width and height is used to reisze the mask image with the product image
        segment(fcn, contentName, height, width)
        
        img = run_mask_transfer('Products/100-stylephoto.jpg', contentName, 'Masks/auto.jpg')
        plt.imshow(img)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()