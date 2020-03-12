from PyQT5.QtWidgets import QApplication, QWidget, QLabel

if __name__ == "__main__":
    app = QApplication()

    w = QWidget()
    w.setGeometry(300, 300, 250, 150)
    w.setWindowTitle("My App")

    label = QLabel("Hello World", w)
    label.setToolTip("This is the tooltip")
    label.resize(label.sizeHint())
    label.move(80, 50)

    w.show()
    app.exec_()