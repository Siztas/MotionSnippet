import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ..core.gesture_controller import GestureController, GestureType
import time
import traceback
from typing import List, Tuple

class MainWindow:
    def __init__(self, master, gesture_controller: GestureController):
        self.master = master
        self.gesture_controller = gesture_controller
        self.recording_gesture = False
        self.recorded_frames = []
        
        # 配置主窗口
        master.geometry("1200x800")
        master.minsize(1200, 800)
        
        # 设置窗口关闭事件处理
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 创建主布局
        self.create_layout()
        
        # 设置回调
        self.gesture_controller.set_callbacks(
            on_frame_update=self.update_frame,
            on_gesture_detected=self.update_gesture_status
        )

    def on_closing(self):
        """窗口关闭时的处理"""
        try:
            print("正在关闭程序...")
            
            # 停止手势控制
            if self.gesture_controller.is_running:
                print("停止手势控制...")
                self.gesture_controller.stop()
            
            # 销毁窗口
            print("销毁窗口...")
            self.master.quit()
            self.master.destroy()
            
            # 强制退出程序
            print("程序退出")
            import os
            os._exit(0)
            
        except Exception as e:
            print(f"关闭程序时出错: {e}")
            # 确保程序退出
            import os
            os._exit(1)

    def create_layout(self):
        """创建主布局"""
        # 创建左右分栏
        self.main_frame = ctk.CTkFrame(self.master)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 左侧面板（视频预览和状态）
        self.left_frame = ctk.CTkFrame(self.main_frame)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # 视频预览区域
        self.preview_frame = ctk.CTkFrame(self.left_frame)
        self.preview_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="")
        self.preview_label.pack(fill="both", expand=True)
        
        # 状态信息区域
        self.status_frame = ctk.CTkFrame(self.left_frame)
        self.status_frame.pack(fill="x", padx=5, pady=5)
        
        self.status_label = ctk.CTkLabel(self.status_frame, text="状态: 未启动")
        self.status_label.pack(side="left", padx=5)
        
        self.gesture_label = ctk.CTkLabel(self.status_frame, text="当前手势: 无")
        self.gesture_label.pack(side="right", padx=5)
        
        # 右侧控制面板
        self.right_frame = ctk.CTkFrame(self.main_frame)
        self.right_frame.pack(side="right", fill="both", padx=5, pady=5)
        
        # 基本控制区域
        self.control_frame = ctk.CTkFrame(self.right_frame)
        self.control_frame.pack(fill="x", padx=5, pady=5)
        
        self.start_button = ctk.CTkButton(
            self.control_frame,
            text="启动",
            command=self.toggle_gesture_control
        )
        self.start_button.pack(fill="x", padx=5, pady=5)
        
        self.mouse_control = ctk.CTkCheckBox(
            self.control_frame,
            text="启用鼠标控制模式",
            command=self.toggle_mouse_control
        )
        self.mouse_control.pack(fill="x", padx=5, pady=5)
        
        # 灵敏度设置
        self.sensitivity_frame = ctk.CTkFrame(self.control_frame)
        self.sensitivity_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(self.sensitivity_frame, text="鼠标灵敏度:").pack(side="left", padx=5)
        
        self.sensitivity_slider = ctk.CTkSlider(
            self.sensitivity_frame,
            from_=1,
            to=20,
            number_of_steps=19,
            command=self.update_sensitivity
        )
        self.sensitivity_slider.set(self.gesture_controller.config["mouse_sensitivity"] * 10)
        self.sensitivity_slider.pack(fill="x", padx=5)
        
        # 手势设置区域
        self.gesture_frame = ctk.CTkScrollableFrame(self.right_frame, label_text="手势设置")
        self.gesture_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 添加预设手势
        for gesture_type in GestureType:
            if gesture_type in self.gesture_controller.gesture_actions:
                action = self.gesture_controller.gesture_actions[gesture_type]
                
                gesture_row = ctk.CTkFrame(self.gesture_frame)
                gesture_row.pack(fill="x", padx=5, pady=2)
                
                ctk.CTkLabel(gesture_row, text=action.description).pack(side="left", padx=5)
                
                key_entry = ctk.CTkEntry(gesture_row)
                key_entry.insert(0, ",".join(action.keys))
                key_entry.pack(side="right", padx=5)
        
        # 自定义手势区域
        self.custom_frame = ctk.CTkFrame(self.right_frame)
        self.custom_frame.pack(fill="x", padx=5, pady=5)
        
        self.gesture_name = ctk.CTkEntry(
            self.custom_frame,
            placeholder_text="输入手势名称"
        )
        self.gesture_name.pack(fill="x", padx=5, pady=5)
        
        self.record_button = ctk.CTkButton(
            self.custom_frame,
            text="开始录制",
            command=self.toggle_gesture_recording
        )
        self.record_button.pack(fill="x", padx=5, pady=5)
        
        # 自定义手势列表
        self.custom_list = ctk.CTkScrollableFrame(self.right_frame, label_text="自定义手势")
        self.custom_list.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.update_custom_gesture_list()

    def update_frame(self, frame: np.ndarray):
        """更新视频帧"""
        try:
            # 转换OpenCV图像为PIL图像
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            
            # 调整图像大小以适应预览区域
            preview_width = self.preview_label.winfo_width()
            preview_height = self.preview_label.winfo_height()
            
            if preview_width > 0 and preview_height > 0:
                image = image.resize(
                    (preview_width, preview_height),
                    Image.Resampling.LANCZOS
                )
            
            # 转换为PhotoImage并显示
            photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            
        except Exception as e:
            print(f"更新视频帧时出错: {e}")

    def update_gesture_status(self, gesture: str):
        """更新手势状态"""
        try:
            self.gesture_label.configure(text=f"当前手势: {gesture}")
        except Exception as e:
            print(f"更新手势状态时出错: {e}")

    def toggle_gesture_control(self):
        """切换手势控制的开启/关闭状态"""
        try:
            if not self.gesture_controller.is_running:
                self.gesture_controller.start()
                self.start_button.configure(text="停止")
                self.status_label.configure(text="状态: 运行中")
            else:
                self.gesture_controller.stop()
                self.start_button.configure(text="启动")
                self.status_label.configure(text="状态: 已停止")
        except Exception as e:
            print(f"切换手势控制时出错: {e}")
            self.start_button.configure(text="启动")
            self.status_label.configure(text="状态: 错误")

    def toggle_mouse_control(self):
        """切换鼠标控制模式"""
        try:
            self.gesture_controller.toggle_mouse_control(
                self.mouse_control.get() == 1
            )
        except Exception as e:
            print(f"切换鼠标控制时出错: {e}")

    def update_sensitivity(self, value):
        """更新鼠标灵敏度"""
        try:
            self.gesture_controller.update_config({
                "mouse_sensitivity": float(value) / 10
            })
        except Exception as e:
            print(f"更新灵敏度时出错: {e}")

    def toggle_gesture_recording(self):
        """切换手势录制状态"""
        try:
            if not self.recording_gesture:
                if not self.gesture_name.get():
                    return
                    
                self.recording_gesture = True
                self.recorded_frames = []
                self.record_button.configure(text="录制中...")
                self.record_button.configure(state="disabled")
                self.status_label.configure(text="状态: 正在录制手势...")
                
                # 创建进度条
                self.progress_bar = ctk.CTkProgressBar(self.custom_frame)
                self.progress_bar.pack(fill="x", padx=5, pady=5)
                self.progress_bar.set(0)
                
                # 设置手势录制回调
                self.gesture_controller.set_gesture_record_callback(self.on_gesture_record)
                
                # 开始进度更新
                self.recording_start_time = time.time()
                self.update_recording_progress()
            else:
                self.recording_gesture = False
                self.record_button.configure(text="开始录制")
                self.record_button.configure(state="normal")
                
                # 移除进度条
                if hasattr(self, 'progress_bar'):
                    self.progress_bar.destroy()
                    delattr(self, 'progress_bar')
                
                # 移除手势录制回调
                self.gesture_controller.set_gesture_record_callback(None)
                
                # 保存并更新界面
                self.save_recorded_gesture()
                    
        except Exception as e:
            print(f"切换手势录制时出错: {e}")
            self.record_button.configure(text="开始录制")
            self.record_button.configure(state="normal")
            self.gesture_controller.set_gesture_record_callback(None)

    def on_gesture_record(self, landmarks_data: List[Tuple[float, float]]):
        """接收手势数据"""
        if self.recording_gesture:
            self.recorded_frames.append(landmarks_data)
            print(f"已录制 {len(self.recorded_frames)} 帧手势数据")

    def update_recording_progress(self):
        """更新录制进度"""
        if not self.recording_gesture:
            return
            
        try:
            elapsed_time = time.time() - self.recording_start_time
            progress = min(elapsed_time / self.gesture_controller.config["recording_duration"], 1.0)
            
            if hasattr(self, 'progress_bar'):
                self.progress_bar.set(progress)
            
            if progress < 1.0:
                # 继续更新
                self.master.after(50, self.update_recording_progress)
            else:
                # 录制完成
                self.toggle_gesture_recording()
                
        except Exception as e:
            print(f"更新录制进度时出错: {e}")
            self.toggle_gesture_recording()

    def save_recorded_gesture(self):
        """保存录制的手势"""
        try:
            if not self.recorded_frames:
                print("没有录制数据，无法保存手势")
                return
                
            # 处理录制数据，计算平均值
            avg_landmarks = np.mean(self.recorded_frames, axis=0)
            
            # 保存手势
            gesture_name = self.gesture_name.get()
            if not gesture_name:
                print("手势名称为空，无法保存")
                return
                
            print(f"保存手势: {gesture_name}")
            self.gesture_controller.add_custom_gesture(gesture_name, avg_landmarks.tolist(), [])
            
            # 更新界面
            self.update_custom_gesture_list()
            self.status_label.configure(text=f"状态: 已保存手势 '{gesture_name}'")
            self.gesture_name.delete(0, "end")
            
            print("手势保存完成，界面已更新")
            
        except Exception as e:
            print(f"保存手势时出错: {e}")
            traceback.print_exc()

    def update_custom_gesture_list(self):
        """更新自定义手势列表"""
        try:
            print("开始更新自定义手势列表")
            print(f"当前自定义手势: {list(self.gesture_controller.custom_gestures.keys())}")
            
            # 清除现有项
            for widget in self.custom_list.winfo_children():
                widget.destroy()
            
            # 添加自定义手势
            for name in self.gesture_controller.custom_gestures.keys():
                row = ctk.CTkFrame(self.custom_list)
                row.pack(fill="x", padx=5, pady=2)
                
                # 添加手势名称标签
                label = ctk.CTkLabel(row, text=name)
                label.pack(side="left", padx=5)
                
                # 添加按键输入框
                keys = self.gesture_controller.get_custom_gesture_keys(name)
                key_entry = ctk.CTkEntry(row, width=120)
                key_entry.insert(0, ",".join(keys))
                key_entry.pack(side="left", padx=5)
                
                # 添加保存按钮
                save_button = ctk.CTkButton(
                    row,
                    text="保存按键",
                    width=80,
                    command=lambda n=name, e=key_entry: self.save_gesture_keys(n, e)
                )
                save_button.pack(side="left", padx=5)
                
                # 添加删除按钮
                delete_button = ctk.CTkButton(
                    row,
                    text="删除",
                    width=60,
                    command=lambda n=name: self.delete_custom_gesture(n)
                )
                delete_button.pack(side="right", padx=5)
            
            # 强制更新界面
            self.custom_list.update()
            print("自定义手势列表更新完成")
                
        except Exception as e:
            print(f"更新自定义手势列表时出错: {e}")
            traceback.print_exc()

    def save_gesture_keys(self, name: str, entry: ctk.CTkEntry):
        """保存手势的按键设置"""
        try:
            keys_text = entry.get().strip()
            keys = [k.strip() for k in keys_text.split(",") if k.strip()]
            self.gesture_controller.update_custom_gesture_keys(name, keys)
            self.status_label.configure(text=f"状态: 已更新手势 '{name}' 的按键设置")
            print(f"已更新手势 '{name}' 的按键: {keys}")
        except Exception as e:
            print(f"保存手势按键时出错: {e}")
            traceback.print_exc()

    def delete_custom_gesture(self, name: str):
        """删除自定义手势"""
        try:
            print(f"删除手势: {name}")
            if name in self.gesture_controller.custom_gestures:
                del self.gesture_controller.custom_gestures[name]
                self.update_custom_gesture_list()
                self.status_label.configure(text=f"状态: 已删除手势 '{name}'")
                print("手势删除完成")
        except Exception as e:
            print(f"删除手势时出错: {e}")
            traceback.print_exc() 