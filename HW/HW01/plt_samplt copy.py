from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton  
import sys  
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  
from matplotlib.figure import Figure  
import numpy as np  


class Window(QMainWindow):  
    def __init__(self):  
        super().__init__()  
        title = "在PyQt5中嵌入Matplotlib - www.linuxmi.com"  
        top = 400  
        left = 400  
        width = 1000  
        height = 600  
        self.setWindowTitle(title)  
        self.setGeometry(top, left, width, height)  
        self.MyUI()  
        
    def MyUI(self):  
        canvas = Canvas(self, width=8, height=4)  
        canvas.move(0,0)  
        button = QPushButton("点击我", self)  
        button.move(100, 500)  
        button2 = QPushButton("再次点击我", self)  
        button2.move(350, 500)  

class Canvas(FigureCanvas):  
    def __init__(self, parent = None, width = 5, height = 5, dpi = 100):  
        fig = Figure(figsize=(width, height), dpi=dpi)  
    
        self.axes = fig.add_subplot(111)  
        FigureCanvas.__init__(self, fig)  
        self.setParent(parent)  
        self.plot()  
    def plot(self):  
        x = np.array([50,30,40,20])  
        labels = ["LinuxMi.com", "Debian", "Linux", "Python"]  
        ax = self.figure.add_subplot(111)  
        ax.pie(x, labels=labels)  
        
app = QApplication(sys.argv)  
window = Window()  
window.show()  
app.exec() 