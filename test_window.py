import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel

def main():
    print("创建测试窗口...")
    app = QApplication(sys.argv)
    
    window = QMainWindow()
    window.setWindowTitle("测试窗口")
    window.setMinimumSize(400, 300)
    
    label = QLabel("测试标签")
    window.setCentralWidget(label)
    
    window.show()
    print("窗口已显示")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 