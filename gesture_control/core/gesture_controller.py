import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import keyboard
import mouse
import time
from threading import Thread, Event
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import traceback
import threading
import json
import os
from pathlib import Path

class GestureType(Enum):
    POINT = "point"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    CUSTOM = "custom"

@dataclass
class GestureAction:
    gesture_type: GestureType
    keys: List[str]
    description: str
    is_mouse: bool = False  # 添加标记，区分是否为鼠标操作

class KalmanFilter:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2, initial_value=0.0):
        self.process_variance = process_variance  # 过程噪声方差
        self.measurement_variance = measurement_variance  # 测量噪声方差
        self.estimate = initial_value  # 当前估计值
        self.estimate_error = 1.0  # 估计误差
        
    def update(self, measurement):
        # 预测步骤
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance
        
        # 更新步骤
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = None  # 初始化时不创建hands实例
        self.mp_draw = mp.solutions.drawing_utils
        
        self.cap = None
        self.is_running = False
        self.stop_event = Event()  # 用于控制线程停止
        self.is_mouse_control_enabled = False
        self.thread = None
        
        # 回调函数
        self.on_frame_update: Optional[Callable[[np.ndarray], None]] = None
        self.on_gesture_detected: Optional[Callable[[str], None]] = None
        
        # 手势跟踪状态
        self.prev_hand_landmarks = None
        self.prev_gesture_time = time.time()
        self.prev_mouse_pos = (0, 0)
        self.prev_mouse_time = time.time()
        self.gesture_cooldown = 0
        
        # 默认手势动作映射
        self.gesture_actions: Dict[GestureType, GestureAction] = {
            GestureType.POINT: GestureAction(
                GestureType.POINT,
                ["click"],
                "Point",
                is_mouse=True  # 标记为鼠标操作
            ),
            GestureType.SWIPE_LEFT: GestureAction(
                GestureType.SWIPE_LEFT,
                ["left"],
                "Swipe Left"
            ),
            GestureType.SWIPE_RIGHT: GestureAction(
                GestureType.SWIPE_RIGHT,
                ["right"],
                "Swipe Right"
            ),
            GestureType.SWIPE_UP: GestureAction(
                GestureType.SWIPE_UP,
                ["up"],
                "Swipe Up"
            ),
            GestureType.SWIPE_DOWN: GestureAction(
                GestureType.SWIPE_DOWN,
                ["down"],
                "Swipe Down"
            )
        }
        
        # 鼠标控制状态
        self.is_pointing = False  # 是否处于指点姿势
        self.prev_is_pointing = False  # 上一帧是否是指点姿势
        self.prev_finger_pos = None  # 上一帧指尖位置
        
        # 自定义手势存储 - 修改为存储手势和动作
        self.custom_gestures: Dict[str, Tuple[List[Tuple[float, float]], List[str]]] = {}
        
        # 添加卡尔曼滤波器
        self.kalman_x = KalmanFilter()
        self.kalman_y = KalmanFilter()
        
        # 配置文件路径
        self.config_dir = Path("config")
        self.config_file = self.config_dir / "settings.json"
        self.gestures_file = self.config_dir / "custom_gestures.json"
        
        # 确保配置目录存在
        self.config_dir.mkdir(exist_ok=True)
        
        # 默认配置参数
        self.default_config = {
            "mouse_sensitivity": 2.5,
            "gesture_threshold": 0.8,
            "smoothing_factor": 0.5,
            "gesture_cooldown": 0.5,
            "mouse_update_interval": 0.008,
            "swipe_threshold": 0.05,  # 降低滑动阈值
            "similarity_threshold": 0.85,
            "recording_duration": 3.0,
            "min_movement_threshold": 0.0005,
            "kalman_process_variance": 1e-4,
            "kalman_measurement_variance": 1e-2,
            "max_move_scale": 0.2,
            "screen_margin": 5
        }
        
        # 加载配置
        self.config = self.load_config()
        
        # 加载自定义手势
        self.custom_gestures = self.load_custom_gestures()
        
        # 设置PyAutoGUI安全设置
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0

    def set_callbacks(self, 
                     on_frame_update: Optional[Callable[[np.ndarray], None]] = None,
                     on_gesture_detected: Optional[Callable[[str], None]] = None):
        """设置回调函数"""
        self.on_frame_update = on_frame_update
        self.on_gesture_detected = on_gesture_detected

    def start(self):
        """启动手势控制"""
        if self.is_running:
            return
            
        try:
            print("Initializing Camera...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("无法打开摄像头")
            
            print("Initializing MediaPipe...")
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            print("Starting Processing Thread...")
            self.is_running = True
            self.stop_event.clear()  # 清除停止标志
            self.thread = Thread(target=self._process_frames)
            self.thread.daemon = True  # 设置为守护线程
            self.thread.start()
            
            print("Gesture Controller Started Successfully")
            
        except Exception as e:
            print(f"Error Starting Gesture Controller: {e}")
            self.stop()
            raise

    def stop(self):
        """停止手势控制"""
        print("Stopping Gesture Controller...")
        
        try:
            # 设置停止标志
            self.stop_event.set()
            self.is_running = False
            
            # 等待线程结束
            if self.thread and self.thread.is_alive():
                print("Waiting for Thread to Finish...")
                self.thread.join(timeout=2.0)
            
            # 释放资源
            if self.cap and self.cap.isOpened():
                print("Releasing Camera Resources...")
                self.cap.release()
                self.cap = None
            
            if self.hands:
                print("Releasing MediaPipe Resources...")
                self.hands.close()
                self.hands = None
            
            # 清理其他资源
            self.prev_hand_landmarks = None
            self.prev_gesture_time = time.time()
            self.prev_mouse_time = time.time()
            
            print("Gesture Controller Stopped Successfully")
            
        except Exception as e:
            print(f"Error Stopping Gesture Controller: {e}")
            traceback.print_exc()
        finally:
            self.is_running = False
            self.cap = None
            self.hands = None

    def _process_frames(self):
        """处理视频帧并识别手势"""
        try:
            while not self.stop_event.is_set():  # 使用事件来控制循环
                if not self.cap or not self.cap.isOpened():
                    print("Camera Closed, Thread Exiting")
                    break
                    
                success, image = self.cap.read()
                if not success:
                    print("Failed to Read Video Frame, Thread Exiting")
                    break

                # 转换颜色空间并进行手部检测
                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                try:
                    results = self.hands.process(image_rgb)
                except Exception as e:
                    print(f"Error Processing Hand Detection: {e}")
                    continue  # 跳过这一帧，继续处理下一帧

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    try:
                        # 绘制手部关键点
                        self.mp_draw.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # 收集手势数据
                        landmarks_data = []
                        for landmark in hand_landmarks.landmark:
                            landmarks_data.append((landmark.x, landmark.y))
                        
                        # 处理手势
                        if self.is_mouse_control_enabled:
                            self._handle_mouse_control(hand_landmarks)
                            if self.on_gesture_detected:
                                self.on_gesture_detected("鼠标控制模式")
                        else:
                            self._detect_and_handle_gestures(hand_landmarks)
                        
                        self.prev_hand_landmarks = hand_landmarks
                        
                        # 如果有录制回调，发送手势数据
                        if hasattr(self, 'on_gesture_record') and self.on_gesture_record:
                            self.on_gesture_record(landmarks_data)
                    except Exception as e:
                        print(f"Error Processing Gesture Data: {e}")
                        continue  # 跳过这一帧的手势处理
                else:
                    self.prev_hand_landmarks = None
                    if self.on_gesture_detected:
                        self.on_gesture_detected("No Gesture")

                # 通过回调更新图像
                if self.on_frame_update and not self.stop_event.is_set():
                    try:
                        self.on_frame_update(image)
                    except Exception as e:
                        print(f"Error Updating Preview Image: {e}")
                    
        except Exception as e:
            print(f"Error Processing Video Frames: {e}")
            traceback.print_exc()
        finally:
            print("Video Processing Thread Ended")
            # 清理资源
            if self.cap and self.cap.isOpened():
                self.cap.release()
            if self.hands:
                self.hands.close()

    def _get_hand_size(self, hand_landmarks) -> float:
        """计算手部大小（使用手掌宽度）"""
        # 使用食指根部(5)到小指根部(17)的距离作为手掌宽度
        index_mcp = hand_landmarks.landmark[5]
        pinky_mcp = hand_landmarks.landmark[17]
        
        dx = index_mcp.x - pinky_mcp.x
        dy = index_mcp.y - pinky_mcp.y
        return np.sqrt(dx * dx + dy * dy)

    def _handle_mouse_control(self, hand_landmarks):
        """处理鼠标控制模式"""
        try:
            current_time = time.time()
            
            # 检测是否是指点姿势
            self.prev_is_pointing = self.is_pointing
            self.is_pointing = self._is_pointing_gesture(hand_landmarks)
            
            # 只在指点姿势时控制鼠标
            if not self.is_pointing:
                self.prev_finger_pos = None  # 重置指尖位置
                # 重置卡尔曼滤波器
                self.kalman_x = KalmanFilter(
                    process_variance=self.config["kalman_process_variance"],
                    measurement_variance=self.config["kalman_measurement_variance"]
                )
                self.kalman_y = KalmanFilter(
                    process_variance=self.config["kalman_process_variance"],
                    measurement_variance=self.config["kalman_measurement_variance"]
                )
                return
                
            # 获取当前指尖位置和手部大小
            index_tip = hand_landmarks.landmark[8]  # 食指指尖
            current_finger_pos = (index_tip.x, index_tip.y)
            hand_size = self._get_hand_size(hand_landmarks)
            
            # 检测是否刚切换到指点姿势
            if not self.prev_is_pointing and self.is_pointing:
                self.prev_finger_pos = current_finger_pos  # 初始化上一帧位置
                try:
                    # 获取当前鼠标位置，确保在安全区域内
                    x, y = pyautogui.position()
                    if self._is_safe_position(x, y):
                        pyautogui.click()
                        if self.on_gesture_detected:
                            self.on_gesture_detected("鼠标点击")
                except Exception as e:
                    print(f"点击鼠标时出错: {e}")
                return
                
            # 限制鼠标更新频率
            if current_time - self.prev_mouse_time < self.config["mouse_update_interval"]:
                return
                
            # 如果有上一帧的位置，计算移动
            if self.prev_finger_pos is not None:
                try:
                    # 计算指尖移动的相对距离（相对于手部大小）
                    dx = (current_finger_pos[0] - self.prev_finger_pos[0]) / hand_size
                    dy = (current_finger_pos[1] - self.prev_finger_pos[1]) / hand_size
                    
                    # 使用卡尔曼滤波器平滑移动
                    filtered_dx = self.kalman_x.update(dx)
                    filtered_dy = self.kalman_y.update(dy)
                    
                    # 检查是否超过最小移动阈值
                    if abs(filtered_dx) > self.config["min_movement_threshold"] or abs(filtered_dy) > self.config["min_movement_threshold"]:
                        # 获取当前鼠标位置
                        current_x, current_y = pyautogui.position()
                        screen_width, screen_height = pyautogui.size()
                        margin = self.config["screen_margin"]
                        
                        # 计算鼠标移动距离（应用灵敏度）
                        scale_factor = min(screen_width, screen_height)
                        sensitivity = self.config["mouse_sensitivity"]
                        
                        # 使用相对于手部大小的移动距离
                        move_x = int(filtered_dx * scale_factor * sensitivity)
                        move_y = int(filtered_dy * scale_factor * sensitivity)
                        
                        # 限制单次移动距离（相对于手部大小）
                        max_move = int(scale_factor * self.config["max_move_scale"])
                        move_x = max(min(move_x, max_move), -max_move)
                        move_y = max(min(move_y, max_move), -max_move)
                        
                        # 计算新位置，考虑安全边界
                        new_x = max(margin, min(current_x + move_x, screen_width - margin))
                        new_y = max(margin, min(current_y + move_y, screen_height - margin))
                        
                        # 只在新位置安全时移动鼠标
                        if self._is_safe_position(new_x, new_y):
                            # 使用绝对位置移动，避免累积误差
                            pyautogui.moveTo(new_x, new_y, duration=0.0)
                        
                except Exception as e:
                    print(f"Error Calculating Mouse Movement: {e}")
                    self.prev_finger_pos = current_finger_pos
                    return
            
            # 更新状态
            self.prev_finger_pos = current_finger_pos
            self.prev_mouse_time = current_time
            
        except Exception as e:
            print(f"Error Handling Mouse Control: {e}")
            traceback.print_exc()

    def _is_safe_position(self, x: int, y: int) -> bool:
        """检查鼠标位置是否在安全区域内"""
        try:
            screen_width, screen_height = pyautogui.size()
            margin = self.config["screen_margin"]
            
            # 检查是否在安全边界内
            return (margin <= x <= screen_width - margin and 
                   margin <= y <= screen_height - margin)
        except Exception as e:
            print(f"Error Checking Mouse Position: {e}")
            return False

    def _detect_and_handle_gestures(self, hand_landmarks):
        """检测和处理"""
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.prev_gesture_time < self.config["gesture_cooldown"]:
            return
            
        # 首先检测自定义手势
        if self.custom_gestures:
            # 提取当前手势的关键点
            current_landmarks = []
            for landmark in hand_landmarks.landmark:
                current_landmarks.append((landmark.x, landmark.y))
                
            # 与所有已保存的手势比较
            for name, (template_landmarks, keys) in self.custom_gestures.items():
                similarity = self._calculate_gesture_similarity(
                    current_landmarks,
                    template_landmarks
                )
                
                if similarity > self.config["similarity_threshold"]:
                    print(f"Detected Custom Gesture: {name}, Similarity: {similarity:.2f}")
                    # 触发自定义手势动作
                    self._trigger_custom_gesture(name)
                    self.prev_gesture_time = current_time
                    return  # 识别到自定义手势就返回，不再检测基本手势
        
        # 如果没有识别到自定义手势，则检测基本手势
        if self._is_pointing_gesture(hand_landmarks):
            self._trigger_action(GestureType.POINT)
            self.prev_gesture_time = current_time
        elif self._is_swipe_gesture(hand_landmarks):
            self.prev_gesture_time = current_time

    def _is_pointing_gesture(self, hand_landmarks) -> bool:
        """检测指点手势"""
        # 食指伸直，其他手指弯曲
        index_tip = hand_landmarks.landmark[8]  # 食指指尖
        index_pip = hand_landmarks.landmark[6]  # 食指第二关节
        middle_tip = hand_landmarks.landmark[12]  # 中指指尖
        ring_tip = hand_landmarks.landmark[16]  # 无名指指尖
        pinky_tip = hand_landmarks.landmark[20]  # 小指指尖
        
        # 检查食指是否伸直
        index_straight = index_tip.y < index_pip.y - 0.1  # 增加阈值
        
        # 检查其他手指是否弯曲
        others_bent = all([
            middle_tip.y > index_pip.y,
            ring_tip.y > index_pip.y,
            pinky_tip.y > index_pip.y
        ])
        
        return index_straight and others_bent

    def _is_swipe_gesture(self, hand_landmarks) -> bool:
        """检测滑动手势"""
        if not self.prev_hand_landmarks:
            return False
            
        # 使用多个关键点来判断滑动
        # 8: 食指尖, 12: 中指尖, 16: 无名指尖, 20: 小指尖
        # 0: 手掌根部
        key_points = [0, 8, 12, 16, 20]
        
        # 计算所有关键点的平均移动
        total_dx = 0
        total_dy = 0
        count = 0
        
        for point_id in key_points:
            curr_point = hand_landmarks.landmark[point_id]
            prev_point = self.prev_hand_landmarks.landmark[point_id]
            
            dx = curr_point.x - prev_point.x
            dy = curr_point.y - prev_point.y
            
            total_dx += dx
            total_dy += dy
            count += 1
        
        # 计算平均移动
        avg_dx = total_dx / count
        avg_dy = total_dy / count
        
        # 判断滑动方向
        threshold = self.config["swipe_threshold"]
        
        # 计算移动距离
        movement = np.sqrt(avg_dx * avg_dx + avg_dy * avg_dy)
        
        # 如果移动距离超过阈值
        if movement > threshold:
            # 判断主要移动方向
            if abs(avg_dx) > abs(avg_dy):
                # 水平移动
                if avg_dx > 0:
                    print("Detected Right Swipe Gesture")
                    self._trigger_action(GestureType.SWIPE_RIGHT)
                else:
                    print("Detected Left Swipe Gesture")
                    self._trigger_action(GestureType.SWIPE_LEFT)
            else:
                # 垂直移动
                if avg_dy > 0:
                    print("Detected Down Swipe Gesture")
                    self._trigger_action(GestureType.SWIPE_DOWN)
                else:
                    print("Detected Up Swipe Gesture")
                    self._trigger_action(GestureType.SWIPE_UP)
            return True
            
        return False

    def _get_palm_center(self, hand_landmarks) -> Tuple[float, float]:
        """计算手掌中心点"""
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        return np.mean(x_coords), np.mean(y_coords)

    def _detect_custom_gestures(self, hand_landmarks):
        """检测自定义手势"""
        if not self.custom_gestures:
            return
            
        # 提取当前手势的关键点
        current_landmarks = []
        for landmark in hand_landmarks.landmark:
            current_landmarks.append((landmark.x, landmark.y))
            
        # 与所有已保存的手势比较
        for name, (template_landmarks, keys) in self.custom_gestures.items():
            similarity = self._calculate_gesture_similarity(
                current_landmarks,
                template_landmarks
            )
            
            if similarity > self.config["similarity_threshold"]:
                # 触发自定义手势动作
                self._trigger_custom_gesture(name)
                self.gesture_cooldown = self.config["gesture_cooldown"]
                break

    def _calculate_gesture_similarity(
        self,
        landmarks1: List[Tuple[float, float]],
        landmarks2: List[Tuple[float, float]]
    ) -> float:
        """计算两个手势的相似度"""
        # 使用欧氏距离计算相似度
        landmarks1 = np.array(landmarks1)
        landmarks2 = np.array(landmarks2)
        
        # 归一化处理
        landmarks1 = self._normalize_landmarks(landmarks1)
        landmarks2 = self._normalize_landmarks(landmarks2)
        
        # 计算距离
        distance = np.mean(np.sqrt(np.sum((landmarks1 - landmarks2) ** 2, axis=1)))
        
        # 转换为相似度分数（0-1之间）
        similarity = 1 / (1 + distance)
        return similarity

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """归一化手势关键点"""
        # 移除平移
        center = np.mean(landmarks, axis=0)
        landmarks = landmarks - center
        
        # 缩放到单位大小
        scale = np.max(np.abs(landmarks))
        if scale > 0:
            landmarks = landmarks / scale
            
        return landmarks

    def _trigger_action(self, gesture_type: GestureType):
        """触发手势对应的动作"""
        if gesture_type in self.gesture_actions:
            action = self.gesture_actions[gesture_type]
            
            try:
                if action.is_mouse:
                    # 处理鼠标操作
                    if action.keys[0] == "click":
                        pyautogui.click()
                    elif action.keys[0] == "rightclick":
                        pyautogui.rightClick()
                    elif action.keys[0] == "doubleclick":
                        pyautogui.doubleClick()
                else:
                    # 同时按下所有按键
                    for key in action.keys:
                        keyboard.press(key)
                    # 短暂延迟以确保按键被识别
                    time.sleep(0.1)
                    # 同时释放所有按键
                    for key in action.keys:
                        keyboard.release(key)
            except Exception as e:
                print(f"Error Executing Action: {e}")
            
            if self.on_gesture_detected:
                self.on_gesture_detected(f"{action.description} {','.join(action.keys)}")

    def _trigger_custom_gesture(self, gesture_name: str):
        """触发自定义手势动作"""
        try:
            if gesture_name not in self.custom_gestures:
                print(f"Custom Gesture Not Found: {gesture_name}")
                return
                
            landmarks, keys = self.custom_gestures[gesture_name]
            if not keys:
                print(f"Custom Gesture '{gesture_name}' has no keys set")
                return
                
            print(f"Triggering Custom Gesture '{gesture_name}' with keys: {keys}")
            try:
                # 检查是否是鼠标操作
                if keys[0] in ["click", "rightclick", "doubleclick"]:
                    if keys[0] == "click":
                        pyautogui.click()
                    elif keys[0] == "rightclick":
                        pyautogui.rightClick()
                    elif keys[0] == "doubleclick":
                        pyautogui.doubleClick()
                else:
                    # 同时按下所有按键
                    for key in keys:
                        keyboard.press(key)
                    # 短暂延迟以确保按键被识别
                    time.sleep(0.1)
                    # 同时释放所有按键
                    for key in keys:
                        keyboard.release(key)
                print(f"Executing Action: {keys}")
            except Exception as e:
                print(f"Error Executing Action: {e}")
            
            if self.on_gesture_detected:
                self.on_gesture_detected(f"Custom Gesture: {gesture_name} ({','.join(keys)})")
                
        except Exception as e:
            print(f"Error Triggering Custom Gesture: {e}")
            traceback.print_exc()

    def add_custom_gesture(self, name: str, landmarks: List[Tuple[float, float]], keys: List[str] = None):
        """添加自定义手势并保存"""
        if keys is None:
            keys = []
        self.custom_gestures[name] = (landmarks, keys)
        self.save_custom_gestures()

    def update_custom_gesture_keys(self, name: str, keys: List[str]):
        """更新自定义手势的按键组合并保存"""
        if name in self.custom_gestures:
            landmarks, _ = self.custom_gestures[name]
            self.custom_gestures[name] = (landmarks, keys)
            self.save_custom_gestures()

    def remove_custom_gesture(self, name: str):
        """删除自定义手势并保存"""
        if name in self.custom_gestures:
            del self.custom_gestures[name]
            self.save_custom_gestures()

    def reset_to_default(self):
        """重置所有设置为默认值"""
        self.config = self.default_config.copy()
        self.save_config()
        self.custom_gestures.clear()
        self.save_custom_gestures()

    def load_config(self) -> dict:
        """从文件加载配置"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # 合并加载的配置和默认配置，确保所有必要的配置项都存在
                    config = self.default_config.copy()
                    config.update(loaded_config)
                    return config
        except Exception as e:
            print(f"Error Loading Config File: {e}")
        return self.default_config.copy()

    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            print("Config Saved")
        except Exception as e:
            print(f"Error Saving Config File: {e}")

    def load_custom_gestures(self) -> Dict[str, Tuple[List[Tuple[float, float]], List[str]]]:
        """从文件加载自定义手势"""
        try:
            if self.gestures_file.exists():
                with open(self.gestures_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 将加载的数据转换为正确的格式
                    gestures = {}
                    for name, gesture_data in data.items():
                        landmarks = [tuple(point) for point in gesture_data['landmarks']]
                        keys = gesture_data['keys']
                        gestures[name] = (landmarks, keys)
                    return gestures
        except Exception as e:
            print(f"Error Loading Custom Gesture File: {e}")
        return {}

    def save_custom_gestures(self):
        """保存自定义手势到文件"""
        try:
            # 将手势数据转换为可序列化的格式
            data = {}
            for name, (landmarks, keys) in self.custom_gestures.items():
                data[name] = {
                    'landmarks': [list(point) for point in landmarks],
                    'keys': keys
                }
            
            with open(self.gestures_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print("Custom Gestures Saved")
        except Exception as e:
            print(f"Error Saving Custom Gesture File: {e}")

    def update_config(self, config: dict):
        """更新配置参数并保存"""
        self.config.update(config)
        self.save_config()

    def toggle_mouse_control(self, enabled: bool):
        """切换鼠标控制模式"""
        self.is_mouse_control_enabled = enabled

    def set_gesture_record_callback(self, callback: Optional[Callable[[List[Tuple[float, float]]], None]]):
        """设置手势录制回调"""
        self.on_gesture_record = callback