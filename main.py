import customtkinter as ctk
import sys
import traceback
import cv2
from gesture_control.ui.main_window import MainWindow
from gesture_control.core.gesture_controller import GestureController

def cleanup():
    """程序退出时的清理工作"""
    print("执行最终清理...")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # 设置主题和默认颜色
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        print("正在初始化手势控制器...")
        controller = GestureController()
        
        print("正在创建主窗口...")
        window = MainWindow(controller)
        
        print("启动应用程序...")
        window.mainloop()
        
    except Exception as e:
        print(f"程序运行时出错: {e}")
        traceback.print_exc()
    finally:
        cleanup()
        sys.exit(0)
 