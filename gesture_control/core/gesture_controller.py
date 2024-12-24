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
from scipy.linalg import orthogonal_procrustes
from enum import Enum
import traceback
import threading
import json
import os
from pathlib import Path

class GestureType(Enum):
    POINT = "point"
    TWO_FINGER_POINT = "two_finger_point"
    THREE_FINGER_POINT = "three_finger_point"
    PINCH = "pinch"
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
    
class ProcrustesTemporalMatcher:
    def __init__(self):
        self.window_size = None  # DTW窗口大小
        self.reference_scale = None  # 参考尺度
        
    def procrustes_distance(self, shape1: np.ndarray, shape2: np.ndarray) -> float:
        """计算两个形状之间的Procrustes距离"""
        # 中心化
        shape1_centered = shape1 - np.mean(shape1, axis=0)
        shape2_centered = shape2 - np.mean(shape2, axis=0)
        
        # 归一化尺度
        scale1 = np.sqrt(np.sum(shape1_centered ** 2))
        scale2 = np.sqrt(np.sum(shape2_centered ** 2))
        shape1_normalized = shape1_centered / scale1
        shape2_normalized = shape2_centered / scale2
        
        # 计算最优旋转矩阵
        R, _ = orthogonal_procrustes(shape1_normalized, shape2_normalized)
        
        # 应用旋转
        shape2_aligned = shape2_normalized @ R
        
        # 计算距离
        return np.mean(np.sqrt(np.sum((shape1_normalized - shape2_aligned) ** 2, axis=1)))
    
    def dtw_procrustes(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """使用Procrustes分析的DTW算法"""
        n, m = len(seq1), len(seq2)
        
        # 初始化DTW矩阵
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # 计算窗口大小
        if self.window_size is None:
            self.window_size = int(max(n, m) * 0.1)  # 默认为序列长度的10%
        
        # 填充DTW矩阵
        for i in range(1, n + 1):
            # 定义Sakoe-Chiba带
            window_start = max(1, i - self.window_size)
            window_end = min(m + 1, i + self.window_size + 1)
            
            for j in range(window_start, window_end):
                # 使用Procrustes距离作为帧间距离
                cost = self.procrustes_distance(
                    seq1[i-1],  # 当前帧的骨架点
                    seq2[j-1]   # 当前帧的骨架点
                )
                
                # 动态规划更新
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # 插入
                    dtw_matrix[i, j-1],    # 删除
                    dtw_matrix[i-1, j-1]   # 匹配
                )
        
        return dtw_matrix[n, m]
    
    def get_alignment_path(self, seq1: np.ndarray, seq2: np.ndarray) -> list:
        """获取最优对齐路径"""
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # 记录路径
        path_matrix = np.zeros((n + 1, m + 1, 2), dtype=int)
        
        # 填充DTW矩阵并记录路径
        for i in range(1, n + 1):
            window_start = max(1, i - self.window_size)
            window_end = min(m + 1, i + self.window_size + 1)
            
            for j in range(window_start, window_end):
                cost = self.procrustes_distance(seq1[i-1], seq2[j-1])
                
                # 找到最小代价路径
                candidates = [
                    (dtw_matrix[i-1, j], (i-1, j)),
                    (dtw_matrix[i, j-1], (i, j-1)),
                    (dtw_matrix[i-1, j-1], (i-1, j-1))
                ]
                min_cost, min_path = min(candidates, key=lambda x: x[0])
                
                dtw_matrix[i, j] = cost + min_cost
                path_matrix[i, j] = min_path
        
        # 回溯找到对齐路径
        path = []
        current = (n, m)
        while current != (0, 0):
            path.append(current)
            current = tuple(path_matrix[current])
        
        return path[::-1]
    
    def align_sequences(self, seq1: np.ndarray, seq2: np.ndarray) -> tuple:
        """对齐两个序列"""
        path = self.get_alignment_path(seq1, seq2)
        
        # 创建对齐后的序列
        aligned_seq1 = []
        aligned_seq2 = []
        
        for i, j in path:
            if i > 0 and j > 0:
                aligned_seq1.append(seq1[i-1])
                aligned_seq2.append(seq2[j-1])
        
        return np.array(aligned_seq1), np.array(aligned_seq2)
    
    def compute_similarity(self, seq1: np.ndarray, seq2: np.ndarray, 
                         temporal_weight: float = 0.5) -> float:
        """计算综合相似度"""
        # 计算DTW-Procrustes距离
        dtw_distance = self.dtw_procrustes(seq1, seq2)
        
        # 获取对齐后的序列
        aligned_seq1, aligned_seq2 = self.align_sequences(seq1, seq2)
        
        # 计算时序相关性
        temporal_correlation = self._compute_temporal_correlation(
            aligned_seq1, 
            aligned_seq2
        )
        
        # 综合评分
        similarity = (1 - temporal_weight) * (1 / (1 + dtw_distance)) + \
                    temporal_weight * temporal_correlation
                    
        return similarity
    
    def _compute_temporal_correlation(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """计算时序相关性"""
        # 计算速度序列
        vel1 = np.diff(seq1, axis=0)
        vel2 = np.diff(seq2, axis=0)
        
        # 计算相关系数
        correlation = np.mean([
            np.corrcoef(vel1[:, i], vel2[:, i])[0, 1]
            for i in range(vel1.shape[1])
            if not np.isnan(np.corrcoef(vel1[:, i], vel2[:, i])[0, 1])
        ])
        
        return (correlation + 1) / 2  # 归一化到[0,1]区间

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
                "指点",
                is_mouse=True
            ),
            GestureType.TWO_FINGER_POINT: GestureAction(
                GestureType.TWO_FINGER_POINT,
                ["rightclick"],
                "双指点击",
                is_mouse=True
            ),
            GestureType.THREE_FINGER_POINT: GestureAction(
                GestureType.THREE_FINGER_POINT,
                ["tab"],
                "三指点击"
            ),
            GestureType.PINCH: GestureAction(
                GestureType.PINCH,
                ["win", "tab"],
                "捏合"
            ),
            GestureType.SWIPE_LEFT: GestureAction(
                GestureType.SWIPE_LEFT,
                ["left"],
                "左滑"
            ),
            GestureType.SWIPE_RIGHT: GestureAction(
                GestureType.SWIPE_RIGHT,
                ["right"],
                "右滑"
            ),
            GestureType.SWIPE_UP: GestureAction(
                GestureType.SWIPE_UP,
                ["pageup"],
                "上滑"
            ),
            GestureType.SWIPE_DOWN: GestureAction(
                GestureType.SWIPE_DOWN,
                ["pagedown"],
                "下滑"
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

        self.procrustes_matcher = ProcrustesTemporalMatcher()
        
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
            "gesture_cooldown": 1.0,  # 设置1秒冷却时间
            "mouse_update_interval": 0.008,
            "swipe_threshold": 0.03,  # 降低滑动阈值
            "similarity_threshold": 0.65,
            "recording_duration": 3.0,
            "min_movement_threshold": 0.0005,
            "kalman_process_variance": 1e-4,
            "kalman_measurement_variance": 1e-2,
            "max_move_scale": 0.2,
            "screen_margin": 5,
            "gesture_buffer_size": 15,
            "min_frames_for_gesture": 5
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
                    # 计算指尖移动的相对��离（相对于手部大小）
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
        """检测和处理手势"""
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.prev_gesture_time < self.config["gesture_cooldown"]:
            return
            
        # 首先检测基本手势
        if self._is_pinch_gesture(hand_landmarks):
            self._trigger_action(GestureType.PINCH)
            self.prev_gesture_time = current_time
            return
            
        if self._is_three_finger_point_gesture(hand_landmarks):
            self._trigger_action(GestureType.THREE_FINGER_POINT)
            self.prev_gesture_time = current_time
            return
            
        if self._is_two_finger_point_gesture(hand_landmarks):
            self._trigger_action(GestureType.TWO_FINGER_POINT)
            self.prev_gesture_time = current_time
            return
            
        if self._is_pointing_gesture(hand_landmarks):
            self._trigger_action(GestureType.POINT)
            self.prev_gesture_time = current_time
            return
            
        if self._is_swipe_gesture(hand_landmarks):
            self.prev_gesture_time = current_time
            return
            
        # 然后检测自定义手势
        if self.custom_gestures:
            # 提取当前手势的关键点序列
            current_landmarks = []
            for landmark in hand_landmarks.landmark:
                current_landmarks.append((landmark.x, landmark.y))
            
            # 将当前帧添加到缓冲区
            if not hasattr(self, 'gesture_buffer'):
                self.gesture_buffer = []
            self.gesture_buffer.append(current_landmarks)
            
            # 保持缓冲区大小固定
            buffer_size = self.config["gesture_buffer_size"]
            if len(self.gesture_buffer) > buffer_size:
                self.gesture_buffer = self.gesture_buffer[-buffer_size:]
            
            # 如果缓冲区达到最小帧数，进行手势识别
            if len(self.gesture_buffer) >= self.config["min_frames_for_gesture"]:
                # 将缓冲区数据转换为numpy数组
                gesture_sequence = np.array(self.gesture_buffer)
                
                # 与所有已保存的手势比较
                best_match = None
                best_similarity = 0
                
                for name, (template_landmarks, keys) in self.custom_gestures.items():
                    try:
                        # 将模板转换为序列形式
                        template_sequence = np.array([template_landmarks] * len(gesture_sequence))
                        
                        # 计算相似度
                        similarity = self.procrustes_matcher.compute_similarity(
                            gesture_sequence,
                            template_sequence,
                            temporal_weight=0.5  # 降低时序权重，更注重形状匹配
                        )
                        
                        print(f"手势 '{name}' 的相似度: {similarity:.2f}")  # 添加调试输出
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = name
                            
                    except Exception as e:
                        print(f"计算手势 '{name}' 的相似度时出错: {e}")
                        continue
                
                # 如果找到匹配的手势
                if best_match and best_similarity > self.config["similarity_threshold"]:
                    print(f"检测到自定义手势: {best_match}, 相似度: {best_similarity:.2f}")
                    # 触发自定义手势动作
                    self._trigger_custom_gesture(best_match)
                    self.prev_gesture_time = current_time
                    # 清空缓冲区
                    self.gesture_buffer.clear()
                    return
                
                # 如果缓冲区太大但没有匹配，清除一半的旧数据
                if len(self.gesture_buffer) >= buffer_size:
                    self.gesture_buffer = self.gesture_buffer[buffer_size//2:]
        
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
        # 8: 食指尖, 12: 中指尖, 0: 手掌根部
        key_points = [0, 8, 12]
        
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
            if abs(avg_dx) > abs(avg_dy) * 1.5:  # 增加水平判定的权重
                # 水平移动
                if avg_dx > 0:
                    print("检测到右滑手势")
                    self._trigger_action(GestureType.SWIPE_RIGHT)
                else:
                    print("检测到左滑手势")
                    self._trigger_action(GestureType.SWIPE_LEFT)
            elif abs(avg_dy) > abs(avg_dx):
                # 垂直移动
                if avg_dy > 0:
                    print("检测到下滑手势")
                    self._trigger_action(GestureType.SWIPE_DOWN)
                else:
                    print("检测到上滑手势")
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
        """计算两个手势的相似度（使用Procrustes分析）"""
        # 转换为numpy数组
        landmarks1 = np.array(landmarks1)
        landmarks2 = np.array(landmarks2)
        
        # 归一化处理
        landmarks1 = self._normalize_landmarks(landmarks1)
        landmarks2 = self._normalize_landmarks(landmarks2)
        
        # 使用Procrustes分析计算相似度
        return self.procrustes_matcher.procrustes_distance(landmarks1, landmarks2)

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """归一化手势关键点"""
        # 移除平移（中心化）
        center = np.mean(landmarks, axis=0)
        landmarks = landmarks - center
        
        # 缩放到单位大小
        scale = np.sqrt(np.sum(landmarks ** 2))
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

    def _is_pinch_gesture(self, hand_landmarks) -> bool:
        """检测捏合手势"""
        # 获取所有指尖和指根的关键点
        thumb_tip = hand_landmarks.landmark[4]  # 拇指尖
        index_tip = hand_landmarks.landmark[8]  # 食指尖
        middle_tip = hand_landmarks.landmark[12]  # 中指尖
        ring_tip = hand_landmarks.landmark[16]  # 无名指尖
        pinky_tip = hand_landmarks.landmark[20]  # 小指尖
        
        # 计算所有指尖到拇指尖的距离
        distances = []
        for tip in [index_tip, middle_tip, ring_tip, pinky_tip]:
            dx = tip.x - thumb_tip.x
            dy = tip.y - thumb_tip.y
            distance = np.sqrt(dx * dx + dy * dy)
            distances.append(distance)
        
        # 如果所有距离都小于阈值，认为是捏合手势
        return all(d < 0.1 for d in distances)

    def _is_three_finger_point_gesture(self, hand_landmarks) -> bool:
        """检测三指指点手势"""
        # 获取指尖和指根的关键点
        index_tip = hand_landmarks.landmark[8]  # 食指尖
        index_pip = hand_landmarks.landmark[6]  # 食指第二关节
        middle_tip = hand_landmarks.landmark[12]  # 中指尖
        middle_pip = hand_landmarks.landmark[10]  # 中指第二关节
        ring_tip = hand_landmarks.landmark[16]  # 无名指尖
        ring_pip = hand_landmarks.landmark[14]  # 无名指第二关节
        pinky_tip = hand_landmarks.landmark[20]  # 小指尖
        
        # 检查三指是否伸直
        index_straight = index_tip.y < index_pip.y - 0.1
        middle_straight = middle_tip.y < middle_pip.y - 0.1
        ring_straight = ring_tip.y < ring_pip.y - 0.1
        
        # 检查小指是否弯曲
        pinky_bent = pinky_tip.y > middle_pip.y
        
        return index_straight and middle_straight and ring_straight and pinky_bent

    def _is_two_finger_point_gesture(self, hand_landmarks) -> bool:
        """检测双指指点手势"""
        # 获取指尖和指根的关键点
        index_tip = hand_landmarks.landmark[8]  # 食指尖
        index_pip = hand_landmarks.landmark[6]  # 食指第二关节
        middle_tip = hand_landmarks.landmark[12]  # 中指尖
        middle_pip = hand_landmarks.landmark[10]  # 中指第二关节
        ring_tip = hand_landmarks.landmark[16]  # 无名指尖
        pinky_tip = hand_landmarks.landmark[20]  # 小指尖
        
        # 检查食指和中指是否伸直
        index_straight = index_tip.y < index_pip.y - 0.1
        middle_straight = middle_tip.y < middle_pip.y - 0.1
        
        # 检查其他手指是否弯曲
        others_bent = all([
            ring_tip.y > middle_pip.y,
            pinky_tip.y > middle_pip.y
        ])
        
        return index_straight and middle_straight and others_bent