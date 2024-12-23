import customtkinter as ctk
import time
import numpy as np
from typing import List, Tuple, Optional

class RecordDialog(ctk.CTkToplevel):
    def __init__(self, parent, gesture_controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        self.gesture_controller = gesture_controller
        self.result = None  # 存储录制结果
        self.recorded_landmarks = []  # 存储录制的手势数据
        
        # 设置窗口属性
        self.title("Record Gesture")
        self.geometry("400x500")
        self.resizable(False, False)
        
        # 使窗口模态
        self.transient(parent)
        self.grab_set()  # 确保窗口是模态的
        
        # 创建界面
        self.create_widgets()
        
        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 录制状态
        self.is_recording = False
        self.start_time = 0
        self.progress_value = 0
        
        # 确保窗口显示在前面
        self.lift()
        self.focus_force()
        
    def create_widgets(self):
        """创建界面组件"""
        # 主容器
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 手势名称输入
        name_frame = ctk.CTkFrame(main_frame)
        name_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(name_frame, text="Gesture Name:").pack(side="left", padx=5)
        self.name_entry = ctk.CTkEntry(name_frame)
        self.name_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        # 按键映射输入
        keys_frame = ctk.CTkFrame(main_frame)
        keys_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(keys_frame, text="Mapping Keys:").pack(side="left", padx=5)
        self.keys_entry = ctk.CTkEntry(keys_frame)
        self.keys_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        # 提示标签
        self.hint_label = ctk.CTkLabel(
            main_frame,
            text="Please enter the gesture name and the mapping keys (multiple keys separated by commas), then click the Start Recording button.",
            wraplength=350
        )
        self.hint_label.pack(pady=10)
        
        # 录制进度条
        self.progress_bar = ctk.CTkProgressBar(main_frame)
        self.progress_bar.pack(fill="x", padx=20, pady=10)
        self.progress_bar.set(0)
        
        # 状态标签
        self.status_label = ctk.CTkLabel(main_frame, text="Ready to Record")
        self.status_label.pack(pady=5)
        
        # 按钮区域
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=10)
        
        self.record_button = ctk.CTkButton(
            button_frame,
            text="Start Recording",
            command=self.toggle_recording
        )
        self.record_button.pack(side="left", padx=5, expand=True)
        
        self.cancel_button = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.cancel
        )
        self.cancel_button.pack(side="right", padx=5)
        
    def toggle_recording(self):
        """切换录制状态"""
        if not self.is_recording:
            # 检查输入
            gesture_name = self.name_entry.get().strip()
            if not gesture_name:
                self.hint_label.configure(text="Please enter the gesture name!", text_color="red")
                return
            
            # 开始录制
            self.start_recording()
        else:
            # 停止录制
            self.stop_recording()
            
    def start_recording(self):
        """开始录制"""
        print("开始录制手势...")
        self.is_recording = True
        self.start_time = time.time()
        self.recorded_landmarks = []
        self.progress_value = 0
        
        # 更新UI
        self.record_button.configure(text="Stop Recording")
        self.status_label.configure(text="Recording...")
        self.hint_label.configure(
            text="Please make the gesture in front of the camera, and the system will record the gesture data for 3 seconds.",
            text_color="white"
        )
        
        # 设置手势录制回调
        self.gesture_controller.set_gesture_record_callback(self.on_gesture_frame)
        
        # 启动进度更新
        self.update_progress()
        
    def stop_recording(self):
        """停止录制"""
        print("停止录制手势...")
        self.is_recording = False
        self.gesture_controller.set_gesture_record_callback(None)
        
        # 保存手势
        gesture_name = self.name_entry.get().strip()
        keys = [k.strip() for k in self.keys_entry.get().split(",") if k.strip()]
        
        if self.recorded_landmarks:
            print(f"Collected {len(self.recorded_landmarks)} frames of gesture data")
            # 计算平均手势数据
            avg_landmarks = self.calculate_average_landmarks()
            # 保存手势
            self.gesture_controller.add_custom_gesture(gesture_name, avg_landmarks, keys)
            print(f"Saved gesture '{gesture_name}' successfully")
            self.result = True
            self.destroy()
        else:
            print("No valid gesture data detected")
            self.hint_label.configure(text="No valid gesture data detected, please try again!", text_color="red")
            self.record_button.configure(text="Start Recording")
            self.status_label.configure(text="Recording Failed")
            self.progress_bar.set(0)
            
    def update_progress(self):
        """更新进度条"""
        if self.is_recording:
            elapsed_time = time.time() - self.start_time
            duration = self.gesture_controller.config["recording_duration"]
            
            if elapsed_time >= duration:
                self.stop_recording()
            else:
                self.progress_value = elapsed_time / duration
                self.progress_bar.set(self.progress_value)
                self.after(50, self.update_progress)  # 每50ms更新一次
                
    def on_gesture_frame(self, landmarks: List[Tuple[float, float]]):
        """处理手势帧数据"""
        if self.is_recording:
            print(f"Received gesture frame data: {len(landmarks)} landmarks")
            self.recorded_landmarks.append(landmarks)
            
    def calculate_average_landmarks(self) -> List[Tuple[float, float]]:
        """计算平均手势数据"""
        if not self.recorded_landmarks:
            return []
            
        # 将所有帧的数据转换为numpy数组
        all_frames = np.array(self.recorded_landmarks)
        # 计算平均值
        avg_landmarks = np.mean(all_frames, axis=0)
        # 转换回元组列表
        return [tuple(point) for point in avg_landmarks]
        
    def cancel(self):
        """取消录制"""
        print("Cancel Recording")
        if self.is_recording:
            self.gesture_controller.set_gesture_record_callback(None)
        self.result = False
        self.destroy()
        
    def on_closing(self):
        """处理窗口关闭事件"""
        self.cancel()

def show_record_dialog(parent, gesture_controller) -> bool:
    """显示录制对话框"""
    try:
        dialog = RecordDialog(parent, gesture_controller)
        parent.wait_window(dialog)  # 等待对话框关闭
        return dialog.result
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False 