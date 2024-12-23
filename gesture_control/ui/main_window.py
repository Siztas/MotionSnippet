import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
from .settings_dialog import show_settings_dialog
from ..core.gesture_controller import GestureController

class MainWindow(ctk.CTk):
    def __init__(self, gesture_controller: GestureController):
        super().__init__()
        
        self.gesture_controller = gesture_controller
        
        # 设置窗口属性
        self.title("手势控制")
        self.geometry("1200x800")
        
        # 创建UI组件
        self.create_widgets()
        
        # 设置回调
        self.gesture_controller.set_callbacks(
            on_frame_update=self.update_preview,
            on_gesture_detected=self.update_status
        )
        
        # 显示窗口
        self.show()

    def create_widgets(self):
        """创建UI组件"""
        # 主布局
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 顶部工具栏
        toolbar = ctk.CTkFrame(main_frame)
        toolbar.pack(fill="x", padx=5, pady=5)
        
        # 在工具栏右侧添加设置按钮
        self.settings_button = ctk.CTkButton(
            toolbar,
            text="设置",
            width=80,
            command=self.show_settings
        )
        self.settings_button.pack(side="right", padx=5)
        
        # 左侧面板 - 预览和控制
        left_panel = ctk.CTkFrame(main_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # 预览区域
        preview_frame = ctk.CTkFrame(left_panel)
        preview_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 预览标签（使用空标签，避免显示初始文本）
        self.preview_label = ctk.CTkLabel(preview_frame, text="")
        self.preview_label.pack(fill="both", expand=True)
        
        # 控制按钮区域
        control_frame = ctk.CTkFrame(left_panel)
        control_frame.pack(fill="x", pady=5)
        
        # 启动/停止按钮
        self.start_button = ctk.CTkButton(
            control_frame,
            text="启动",
            command=self.toggle_gesture_control
        )
        self.start_button.pack(side="left", padx=5)
        
        # 鼠标控制开关
        self.mouse_control = ctk.CTkSwitch(
            control_frame,
            text="鼠标控制",
            command=self.toggle_mouse_control
        )
        self.mouse_control.pack(side="left", padx=10)
        
        # 状态标签
        self.status_label = ctk.CTkLabel(control_frame, text="状态: 未启动")
        self.status_label.pack(side="right", padx=5)
        
        # 右侧面板 - 手势列表和自定义
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", padx=5, pady=5)
        
        # 手势列表标签
        ctk.CTkLabel(right_panel, text="可用手势:").pack(anchor="w", padx=5, pady=5)
        
        # 手势列表
        gestures_frame = ctk.CTkFrame(right_panel)
        gestures_frame.pack(fill="x", padx=5, pady=5)
        
        # 添加默认手势
        self.add_gesture_item(gestures_frame, "指点", "click")
        self.add_gesture_item(gestures_frame, "左滑", "left")
        self.add_gesture_item(gestures_frame, "右滑", "right")
        self.add_gesture_item(gestures_frame, "上滑", "up")
        self.add_gesture_item(gestures_frame, "下滑", "down")
        
        # 自定义手势区域
        custom_frame = ctk.CTkFrame(right_panel)
        custom_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(custom_frame, text="自定义手势").pack(anchor="w", padx=5, pady=5)
        
        self.record_button = ctk.CTkButton(
            custom_frame,
            text="录制新手势",
            command=self.start_recording
        )
        self.record_button.pack(fill="x", padx=5, pady=5)

    def add_gesture_item(self, parent, name: str, action: str):
        """添加手势项"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(frame, text=f"{name} ({action})").pack(side="left", padx=5)

    def toggle_gesture_control(self):
        """切换手势控制状态"""
        if not self.gesture_controller.is_running:
            try:
                self.gesture_controller.start()
                self.start_button.configure(text="停止")
                self.status_label.configure(text="状态: 运行中")
            except Exception as e:
                self.status_label.configure(text=f"启动失败: {str(e)}")
        else:
            self.gesture_controller.stop()
            self.start_button.configure(text="启动")
            self.status_label.configure(text="状态: 已停止")
            # 清除预览图像
            self.preview_label.configure(image=None)

    def toggle_mouse_control(self):
        """切换鼠标控制模式"""
        enabled = self.mouse_control.get()
        self.gesture_controller.toggle_mouse_control(enabled)
        if enabled:
            self.status_label.configure(text="状态: 鼠标控制已启用")
        else:
            self.status_label.configure(text="状态: 鼠标控制已禁用")

    def show_settings(self):
        """显示设置对话框"""
        if show_settings_dialog(self, self.gesture_controller):
            # 如果设置已更改，更新UI
            self.status_label.configure(text="设置已更新")

    def update_preview(self, frame: np.ndarray):
        """更新预览画面"""
        if frame is None:
            return
            
        # 调整图像大小
        height, width = frame.shape[:2]
        max_size = 640
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # 转换为PIL图像
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        
        # 更新标签
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo  # 保持引用

    def update_status(self, status: str):
        """更新状态显示"""
        self.status_label.configure(text=f"状态: {status}")

    def start_recording(self):
        """开始录制新手势"""
        # TODO: 实现手势录制功能
        pass

    def show(self):
        """显示窗口"""
        self.deiconify()  # 确保窗口可见
        self.lift()  # 将窗口提升到顶层
        self.focus_force()  # 强制获取焦点