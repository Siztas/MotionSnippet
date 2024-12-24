import customtkinter as ctk
import cv2
import screeninfo
from typing import Dict, Any, Optional, Callable

class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, gesture_controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        self.gesture_controller = gesture_controller
        self.result = None  # 存储设置结果
        self.was_running = self.gesture_controller.is_running  # 记录原始运行状态
        
        # 设置窗口属性
        self.title("Settings")
        self.geometry("600x800")
        self.resizable(False, False)
        
        # 使窗口模态，但不阻塞主窗口的视频处理
        self.transient(parent)
        
        # 创建设置界面
        self.create_widgets()
        
        # 加载当前设置
        self.load_current_settings()
        
        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 保存原始配置
        self._original_config = self.gesture_controller.config.copy()

    def on_closing(self):
        """处理窗口关闭事件"""
        try:
            # 如果取消设置，恢复原始配置
            if not self.result and hasattr(self, '_original_config'):
                # 只更新非鼠标灵敏度的设置
                restore_config = {k: v for k, v in self._original_config.items() 
                               if k != "mouse_sensitivity"}
                if restore_config:
                    self.gesture_controller.update_config(restore_config)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.destroy()

    def show(self):
        """显示设置对话框"""
        # 显示窗口
        self.deiconify()
        self.focus_force()
        
        # 等待窗口关闭
        self.wait_window()
        return self.result

    def create_widgets(self):
        """创建设置界面组件"""
        # 创建滚动容器
        self.scrollable_frame = ctk.CTkScrollableFrame(self)
        self.scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 摄像头设置
        self.create_camera_settings()
        
        # 显示器设置
        self.create_display_settings()
        
        # 手势阈值设置
        self.create_gesture_settings()
        
        # 鼠标控制设置
        self.create_mouse_settings()
        
        # 按钮区域
        self.create_button_area()

    def create_camera_settings(self):
        """创建摄像头设置区域"""
        camera_frame = ctk.CTkFrame(self.scrollable_frame)
        camera_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(camera_frame, text="Camera Settings").pack(anchor="w", padx=5, pady=5)
        
        # 摄像头选择
        camera_select_frame = ctk.CTkFrame(camera_frame)
        camera_select_frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(camera_select_frame, text="Select Camera:").pack(side="left", padx=5)
        
        # 获取可用摄像头列表
        available_cameras = self.get_available_cameras()
        
        # 创建摄像头选择变量并设置默认值
        self.camera_var = ctk.StringVar(value="Camera 0")  # 默认使用摄像头0
        
        # 创建下拉菜单
        camera_menu = ctk.CTkOptionMenu(
            camera_select_frame,
            variable=self.camera_var,
            values=available_cameras
        )
        camera_menu.pack(side="left", padx=5)
        
        # 分辨率设置
        resolution_frame = ctk.CTkFrame(camera_frame)
        resolution_frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(resolution_frame, text="Resolution:").pack(side="left", padx=5)
        
        resolutions = [
            "640x480",
            "800x600",
            "1280x720",
            "1920x1080"
        ]
        
        self.resolution_var = ctk.StringVar(value="640x480")  # 默认分辨率
        
        resolution_menu = ctk.CTkOptionMenu(
            resolution_frame,
            variable=self.resolution_var,
            values=resolutions
        )
        resolution_menu.pack(side="left", padx=5)

    def create_display_settings(self):
        """创建显示器设置区域"""
        display_frame = ctk.CTkFrame(self.scrollable_frame)
        display_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(display_frame, text="Display Settings").pack(anchor="w", padx=5, pady=5)
        
        # 获取可用显示器列表
        available_displays = self.get_available_displays()
        
        # 显示器选择下拉框
        self.display_var = ctk.StringVar()
        self.display_dropdown = ctk.CTkOptionMenu(
            display_frame,
            variable=self.display_var,
            values=available_displays
        )
        self.display_dropdown.pack(fill="x", padx=5, pady=5)

    def create_gesture_settings(self):
        """创建手势设置区域"""
        gesture_frame = ctk.CTkFrame(self.scrollable_frame)
        gesture_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(gesture_frame, text="Gesture Settings").pack(anchor="w", padx=5, pady=5)
        
        # 手势检测阈值
        self.create_slider(
            gesture_frame,
            "Gesture Detection Threshold:",
            "gesture_threshold",
            0.5, 1.0, 0.05
        )
        
        # 手势相似度阈值
        self.create_slider(
            gesture_frame,
            "Gesture Similarity Threshold:",
            "similarity_threshold",
            0.5, 1.0, 0.05
        )
        
        # 滑动手势阈值
        self.create_slider(
            gesture_frame,
            "Swipe Gesture Threshold:",
            "swipe_threshold",
            0.05, 0.2, 0.01
        )
        
        # 手势冷却时间
        self.create_slider(
            gesture_frame,
            "Gesture Cooldown Time(seconds):",
            "gesture_cooldown",
            0.1, 2.0, 0.1
        )

    def create_mouse_settings(self):
        """创建鼠标控制设置区域"""
        mouse_frame = ctk.CTkFrame(self.scrollable_frame)
        mouse_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(mouse_frame, text="Mouse Control Settings").pack(anchor="w", padx=5, pady=5)
        
        # 鼠标灵敏度 - 使用实时更新
        sensitivity_frame = ctk.CTkFrame(mouse_frame)
        sensitivity_frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(sensitivity_frame, text="Mouse Sensitivity:").pack(side="left", padx=5)
        
        self.sensitivity_label = ctk.CTkLabel(sensitivity_frame, text="0.0")
        self.sensitivity_label.pack(side="right", padx=5)
        
        self.sensitivity_slider = ctk.CTkSlider(
            sensitivity_frame,
            from_=0.5,
            to=3.0,
            number_of_steps=25,
            command=self.update_mouse_sensitivity
        )
        self.sensitivity_slider.pack(side="left", fill="x", expand=True, padx=5)
        
        # 设置初始值
        initial_sensitivity = self.gesture_controller.config.get("mouse_sensitivity", 1.5)
        self.sensitivity_slider.set(initial_sensitivity)
        self.sensitivity_label.configure(text=f"{initial_sensitivity:.2f}")
        
        # 最小移动阈值
        self.create_slider(
            mouse_frame,
            "Minimum Movement Threshold:",
            "min_movement_threshold",
            0.0001, 0.01, 0.0001
        )
        
        # 鼠标更新间隔
        self.create_slider(
            mouse_frame,
            "Mouse Update Interval(seconds):",
            "mouse_update_interval",
            0.016, 0.1, 0.001
        )

    def update_mouse_sensitivity(self, value):
        """实时更新鼠标灵敏度"""
        try:
            sensitivity = float(value)
            self.sensitivity_label.configure(text=f"{sensitivity:.2f}")
            # 直接更新手势控制器的配置
            self.gesture_controller.update_config({"mouse_sensitivity": sensitivity})
        except Exception as e:
            print(f"更新鼠标灵敏度时出错: {e}")

    def create_button_area(self):
        """创建按钮区域"""
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        # 保存按钮
        save_button = ctk.CTkButton(
            button_frame,
            text="Save",
            command=self.save_settings
        )
        save_button.pack(side="left", padx=5)
        
        # 取消按钮
        cancel_button = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.cancel
        )
        cancel_button.pack(side="right", padx=5)
        
        # 重置按钮
        reset_button = ctk.CTkButton(
            button_frame,
            text="Reset to Default",
            command=self.reset_to_default
        )
        reset_button.pack(side="right", padx=5)

    def create_slider(self, parent, label: str, key: str, from_: float, to: float, step: float):
        """创建滑动条设置项"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(frame, text=label).pack(side="left", padx=5)
        
        slider = ctk.CTkSlider(
            frame,
            from_=from_,
            to=to,
            number_of_steps=int((to - from_) / step)
        )
        slider.pack(side="left", fill="x", expand=True, padx=5)
        
        value_label = ctk.CTkLabel(frame, text="0.0")
        value_label.pack(side="right", padx=5)
        
        # 保存引用
        if not hasattr(self, "sliders"):
            self.sliders = {}
        self.sliders[key] = (slider, value_label)
        
        # 更新数值显示
        def update_value(value):
            value_label.configure(text=f"{float(value):.3f}")
        
        slider.configure(command=update_value)

    def get_available_cameras(self) -> list:
        """获取可用摄像头列表"""
        available_cameras = []
        for i in range(10):  # 检查前10个摄像头索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(f"Camera {i}")
                cap.release()
        return available_cameras if available_cameras else ["Default Camera"]

    def get_available_displays(self) -> list:
        """获取可用显示器列表"""
        try:
            monitors = screeninfo.get_monitors()
            return [f"Display {i+1} ({m.width}x{m.height})" for i, m in enumerate(monitors)]
        except:
            return ["Main Display"]

    def load_current_settings(self):
        """加载当前设置"""
        # 加载滑动条设置
        for key, (slider, label) in self.sliders.items():
            if key in self.gesture_controller.config:
                value = self.gesture_controller.config[key]
                slider.set(value)
                label.configure(text=f"{value:.3f}")
        
        # 加载摄像头设置
        if "camera_index" in self.gesture_controller.config:
            camera_index = self.gesture_controller.config["camera_index"]
            self.camera_var.set(f"Camera {camera_index}")
        else:
            self.camera_var.set("Camera 0")  # 默认使用摄像头0
        
        # 加载分辨率设置
        if "camera_resolution" in self.gesture_controller.config:
            width, height = self.gesture_controller.config["camera_resolution"]
            self.resolution_var.set(f"{width}x{height}")
        else:
            self.resolution_var.set("640x480")  # 默认分辨率

    def save_settings(self):
        """保存设置"""
        try:
            # 收集所有设置（除了鼠标灵敏度，因为它是实时更新的）
            new_config = {}
            need_restart = False
            
            # 1. 获取滑动条值（排除鼠标灵敏度）
            for key, (slider, _) in self.sliders.items():
                if key != "mouse_sensitivity":  # 排除鼠标灵敏度
                    new_config[key] = float(slider.get())
            
            # 2. 处理摄像头设置
            camera_value = self.camera_var.get()
            if camera_value and camera_value != "Default Camera":
                try:
                    camera_parts = camera_value.split()
                    if len(camera_parts) >= 2 and camera_parts[0] == "Camera":
                        camera_index = int(camera_parts[1])
                        if "camera_index" not in self.gesture_controller.config or \
                           self.gesture_controller.config["camera_index"] != camera_index:
                            new_config["camera_index"] = camera_index
                            need_restart = True
                except (ValueError, IndexError) as e:
                    print(f"解析摄像头索引时出错: {e}")
            
            # 3. 处理分辨率设置
            resolution = self.resolution_var.get()
            if resolution:
                try:
                    width, height = map(int, resolution.split("x"))
                    new_resolution = (width, height)
                    if "camera_resolution" not in self.gesture_controller.config or \
                       self.gesture_controller.config["camera_resolution"] != new_resolution:
                        new_config["camera_resolution"] = new_resolution
                        need_restart = True
                except (ValueError, AttributeError) as e:
                    print(f"解析分辨率时出错: {e}")
            
            # 4. 处理显示器设置
            display = self.display_var.get()
            if display and display != "Main Display":
                try:
                    display_parts = display.split()
                    if len(display_parts) >= 2:
                        display_index = int(display_parts[1].split("(")[0]) - 1
                        if "display_index" not in self.gesture_controller.config or \
                           self.gesture_controller.config["display_index"] != display_index:
                            new_config["display_index"] = display_index
                except (ValueError, IndexError) as e:
                    print(f"解析显示器索引时出错: {e}")
            
            # 5. 更新配置（除了鼠标灵敏度）
            if new_config:
                self.gesture_controller.update_config(new_config)
            
            # 6. 如果需要，重启摄像头
            if need_restart and self.gesture_controller.is_running:
                print("Restart Camera to Apply New Settings...")
                was_running = self.gesture_controller.is_running
                if was_running:
                    self.gesture_controller.stop()
                self.gesture_controller.update_config(new_config)
                if was_running:
                    self.gesture_controller.start()
            
            self.result = True
            self.destroy()
            
        except Exception as e:
            print(f"Error Saving Settings: {e}")
            import traceback
            traceback.print_exc()

    def cancel(self):
        """取消设置"""
        self.result = False
        self.destroy()

    def reset_to_default(self):
        """重置为默认设置"""
        default_config = {
            "mouse_sensitivity": 1.5,
            "gesture_threshold": 0.8,
            "smoothing_factor": 0.5,
            "gesture_cooldown": 0.5,
            "mouse_update_interval": 0.016,
            "swipe_threshold": 0.1,
            "similarity_threshold": 0.85,
            "min_movement_threshold": 0.001,
            "camera_index": 0,  # 默认使用摄像头0
            "camera_resolution": (640, 480)  # 默认分辨率
        }
        
        # 更新滑动条
        for key, (slider, label) in self.sliders.items():
            if key in default_config:
                value = default_config[key]
                slider.set(value)
                label.configure(text=f"{value:.3f}")
        
        # 更新摄像头选择
        self.camera_var.set("Camera 0")
        
        # 更新分辨率
        self.resolution_var.set("640x480")

def show_settings_dialog(parent, gesture_controller) -> bool:
    """显示设置对话框"""
    try:
        # 创建并显示对话框
        dialog = SettingsDialog(parent, gesture_controller)
        return dialog.show()
    except Exception as e:
        print(f"Error Showing Settings Dialog: {e}")
        import traceback
        traceback.print_exc()
        return False 