import customtkinter as ctk
import sys
import traceback
import atexit
import cv2
from gesture_control.ui.main_window import MainWindow
from gesture_control.core.gesture_controller import GestureController

def cleanup():
    """程序退出时的清理工作"""
    print("执行最终清理...")
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == "__main__":
    try:
        # 注册退出处理函数
        atexit.register(cleanup)
        
        # 设置主题和默认颜色
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # 创建应用程序
        app = ctk.CTk()
        app.title("Gesture Control Pro")
        
        # 创建手势控制器
        controller = GestureController()
        
        # 创建主窗口
        window = MainWindow(app, controller)
        
        # 启动应用程序
        app.mainloop()
        
    except Exception as e:
        print(f"程序运行时出错: {e}")
        traceback.print_exc()
        sys.exit(1)
 