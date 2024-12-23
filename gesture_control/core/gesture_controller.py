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
                is_mouse=True  # 标记为鼠标操作
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
                ["up"],
                "上滑"
            ),
            GestureType.SWIPE_DOWN: GestureAction(
                GestureType.SWIPE_DOWN,
                ["down"],
                "下滑"
            )
        }
        
        # 鼠标控制状态
        self.is_pointing = False  # 是否处于指点姿势
        self.prev_is_pointing = False  # 上一帧是否是指点姿势
        self.prev_finger_pos = None  # 上一帧指尖位置
        
        # 自定义手势存储 - 修改为存储手势和动作
        self.custom_gestures: Dict[str, Tuple[List[Tuple[float, float]], List[str]]] = {}
        
        # 配置参数
        self.config = {
            "mouse_sensitivity": 1.5,  # 鼠标移动灵敏度
            "gesture_threshold": 0.8,
            "smoothing_factor": 0.5,
            "gesture_cooldown": 0.5,  # 秒
            "mouse_update_interval": 0.016,  # 约60fps
            "swipe_threshold": 0.1,  # 归一化距离
            "similarity_threshold": 0.85,  # 手势相似度阈值
            "recording_duration": 3.0,  # 录制持续时间（秒）
            "min_movement_threshold": 0.001  # 最小移动阈值，防止抖动
        }

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
            print("初始化摄像头...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("无法打开摄像头")
            
            print("初始化MediaPipe...")
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            print("启动处理线程...")
            self.is_running = True
            self.stop_event.clear()  # 清除停止标志
            self.thread = Thread(target=self._process_frames)
            self.thread.daemon = True  # 设置为守护线程
            self.thread.start()
            
            print("手势控制器启动成功")
            
        except Exception as e:
            print(f"启动手势控制器时出错: {e}")
            self.stop()
            raise

    def stop(self):
        """停止手势控制"""
        print("正在停止手势控制器...")
        
        try:
            # 设置停止标志
            self.stop_event.set()
            self.is_running = False
            
            # 等待线程结束
            if self.thread and self.thread.is_alive():
                print("等待线程结束...")
                self.thread.join(timeout=2.0)
            
            # 释放资源
            if self.cap and self.cap.isOpened():
                print("释放摄像头资源...")
                self.cap.release()
                self.cap = None
            
            if self.hands:
                print("释放MediaPipe资源...")
                self.hands.close()
                self.hands = None
            
            # 清理其他资源
            self.prev_hand_landmarks = None
            self.prev_gesture_time = time.time()
            self.prev_mouse_time = time.time()
            
            print("手势控制器已停止")
            
        except Exception as e:
            print(f"停止手势控制器时出错: {e}")
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
                    print("摄像头已关闭，线程退出")
                    break
                    
                success, image = self.cap.read()
                if not success:
                    print("读取视频帧失败，线程退出")
                    break

                # 转换颜色空间并进行手部检测
                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                try:
                    results = self.hands.process(image_rgb)
                except Exception as e:
                    print(f"处理手部检测时出错: {e}")
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
                        print(f"处理手势数据时出错: {e}")
                        continue  # 跳过这一帧的手势处理
                else:
                    self.prev_hand_landmarks = None
                    if self.on_gesture_detected:
                        self.on_gesture_detected("无手势")

                # 通过回���更新图像
                if self.on_frame_update and not self.stop_event.is_set():
                    try:
                        self.on_frame_update(image)
                    except Exception as e:
                        print(f"更新预览图像时出错: {e}")
                    
        except Exception as e:
            print(f"处理视频帧时出错: {e}")
            traceback.print_exc()
        finally:
            print("视频处理线程结束")
            # 清理资源
            if self.cap and self.cap.isOpened():
                self.cap.release()
            if self.hands:
                self.hands.close()

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
                return
                
            # 获取当前指尖位置
            index_tip = hand_landmarks.landmark[8]  # 食指指尖
            current_finger_pos = (index_tip.x, index_tip.y)
            
            # 检测是否刚切换到指点姿势
            if not self.prev_is_pointing and self.is_pointing:
                self.prev_finger_pos = current_finger_pos  # 初始化上一帧位置
                try:
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
                    # 计算指尖移动的相对距离
                    dx = current_finger_pos[0] - self.prev_finger_pos[0]
                    dy = current_finger_pos[1] - self.prev_finger_pos[1]
                    
                    # 检查是否超过最小移动阈值
                    if abs(dx) > self.config["min_movement_threshold"] or abs(dy) > self.config["min_movement_threshold"]:
                        # 获取当前鼠标位置
                        current_x, current_y = pyautogui.position()
                        
                        # 计算鼠标移动距离（应用灵敏度）
                        # 将移动距离缩放到合适的范围（屏幕分辨率的10%）
                        screen_width, screen_height = pyautogui.size()
                        scale_factor = min(screen_width, screen_height) * 0.1
                        sensitivity = self.config["mouse_sensitivity"]
                        
                        move_x = int(dx * scale_factor * sensitivity)
                        move_y = int(dy * scale_factor * sensitivity)
                        
                        # 限制单次移动距离
                        max_move = 100  # 最大移动100像素
                        move_x = max(min(move_x, max_move), -max_move)
                        move_y = max(min(move_y, max_move), -max_move)
                        
                        # 计算新位置
                        new_x = max(0, min(current_x + move_x, screen_width - 1))
                        new_y = max(0, min(current_y + move_y, screen_height - 1))
                        
                        # 使用相对移动而不是绝对位置
                        pyautogui.moveRel(move_x, move_y, duration=0.01)
                        
                except Exception as e:
                    print(f"计算鼠标移动时出错: {e}")
                    self.prev_finger_pos = current_finger_pos
                    return
            
            # 更新状态
            self.prev_finger_pos = current_finger_pos
            self.prev_mouse_time = current_time
            
        except Exception as e:
            print(f"鼠标控制时出错: {e}")
            traceback.print_exc()

    def _detect_and_handle_gestures(self, hand_landmarks):
        """检测和处理手势"""
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
                    print(f"检测到自定义手势: {name}, 相似度: {similarity:.2f}")
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
            
        # 计算手掌中心点的移动
        current_center = self._get_palm_center(hand_landmarks)
        prev_center = self._get_palm_center(self.prev_hand_landmarks)
        
        dx = current_center[0] - prev_center[0]
        dy = current_center[1] - prev_center[1]
        
        # 判断滑动方向
        threshold = self.config["swipe_threshold"]
        if abs(dx) > threshold or abs(dy) > threshold:
            if abs(dx) > abs(dy):
                if dx > 0:
                    self._trigger_action(GestureType.SWIPE_RIGHT)
                else:
                    self._trigger_action(GestureType.SWIPE_LEFT)
            else:
                if dy > 0:
                    self._trigger_action(GestureType.SWIPE_DOWN)
                else:
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
            
            for key in action.keys:
                try:
                    if action.is_mouse:
                        # 处理鼠标操作
                        if key == "click":
                            pyautogui.click()
                        elif key == "rightclick":
                            pyautogui.rightClick()
                        elif key == "doubleclick":
                            pyautogui.doubleClick()
                        # 可以添加更多鼠标操作...
                    else:
                        # 处理键盘操作
                        keyboard.press_and_release(key)
                except Exception as e:
                    print(f"执行动作时出错: {e}")
            
            if self.on_gesture_detected:
                self.on_gesture_detected(f"{action.description} {','.join(action.keys)}")

    def _trigger_custom_gesture(self, gesture_name: str):
        """触发自定义手势动作"""
        try:
            if gesture_name not in self.custom_gestures:
                print(f"未找到自定义手势: {gesture_name}")
                return
                
            landmarks, keys = self.custom_gestures[gesture_name]
            if not keys:
                print(f"自定义手势 '{gesture_name}' 没有设置按键")
                return
                
            print(f"触发自定义手势 '{gesture_name}' 的按键: {keys}")
            for key in keys:
                try:
                    # 检查是否是鼠标操作
                    if key in ["click", "rightclick", "doubleclick"]:
                        if key == "click":
                            pyautogui.click()
                        elif key == "rightclick":
                            pyautogui.rightClick()
                        elif key == "doubleclick":
                            pyautogui.doubleClick()
                    else:
                        keyboard.press_and_release(key)
                    print(f"执行操作: {key}")
                except Exception as e:
                    print(f"执行操作 {key} 时出错: {e}")
            
            if self.on_gesture_detected:
                self.on_gesture_detected(f"自定义手势: {gesture_name} ({','.join(keys)})")
                
        except Exception as e:
            print(f"触发自定义手势时出错: {e}")
            traceback.print_exc()

    def add_custom_gesture(self, name: str, landmarks: List[Tuple[float, float]], keys: List[str] = None):
        """添加自定义手势"""
        if keys is None:
            keys = []
        self.custom_gestures[name] = (landmarks, keys)

    def update_custom_gesture_keys(self, name: str, keys: List[str]):
        """更新自定义手势的按键组合"""
        if name in self.custom_gestures:
            landmarks, _ = self.custom_gestures[name]
            self.custom_gestures[name] = (landmarks, keys)

    def get_custom_gesture_keys(self, name: str) -> List[str]:
        """获取自定义手势的按键组合"""
        if name in self.custom_gestures:
            return self.custom_gestures[name][1]
        return []

    def update_config(self, config: dict):
        """更新配置参数"""
        self.config.update(config)

    def toggle_mouse_control(self, enabled: bool):
        """切换鼠标控制模式"""
        self.is_mouse_control_enabled = enabled

    def set_gesture_record_callback(self, callback: Optional[Callable[[List[Tuple[float, float]]], None]]):
        """设置手势录制回调"""
        self.on_gesture_record = callback