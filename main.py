import customtkinter as ctk
import sys
import traceback
import cv2
from gesture_control.ui.main_window import MainWindow
from gesture_control.core.gesture_controller import GestureController

def cleanup():
    """程序退出时的清理工作"""
    print("Execute Final Cleanup...")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # 设置主题和默认颜色
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        print("Init...")
        controller = GestureController()
        
        print("Create Main Window...")
        window = MainWindow(controller)
        
        print("Start Application...")
        window.mainloop()
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        cleanup()
        sys.exit(0)
 