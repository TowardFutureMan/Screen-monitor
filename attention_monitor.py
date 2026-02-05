import sys
import os

# ---------- 调试日志：从启动第一行就写入，用于定位 .app 闪退 ----------
# 日志文件固定放在用户主目录，双击 .app 时也能写入
_DEBUG_LOG_PATH = os.path.join(os.path.expanduser("~"), "AttentionMonitor_debug.log")

def _log(msg):
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()
    except Exception:
        pass

_log("=== Attention Monitor 启动 ===")

import tempfile
_log("tempfile ok")

# ==========================================
# 0. 核心环境配置 (必须最先执行)
# ==========================================
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'matplotlib_cache')
_log("MPLCONFIGDIR set")

import traceback

# 确保 multiprocessing 支持
import multiprocessing
multiprocessing.freeze_support()
_log("multiprocessing ok")

import subprocess
_log("subprocess ok")
import cv2
_log("cv2 ok")
import mediapipe as mp
_log("mediapipe ok")
import numpy as np
import time
import pygame
_log("pygame ok")
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
_log("tk/PIL ok")

# 定义一个全局异常捕获函数，用于在 GUI 启动前弹窗报错
def show_critical_error(error_msg):
    try:
        # 使用 AppleScript 弹窗，保证在任何环境下都能看到
        script = f'display dialog "{error_msg}" with title "Attention Monitor Error" buttons {{"OK"}} default button "OK" with icon stop'
        os.system(f"osascript -e '{script}'")
    except:
        pass
    # 同时写入文件
    try:
        with open(os.path.join(os.getcwd(), 'CRITICAL_ERROR.txt'), 'w') as f:
            f.write(error_msg)
    except:
        pass

# ==========================================
# 0. 核心环境配置 (必须最先执行)
# ==========================================
# 强制配置 Matplotlib 缓存目录到临时文件夹
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'matplotlib_cache')

# 确保 multiprocessing 支持
import multiprocessing
multiprocessing.freeze_support()

import subprocess
import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

def resource_path(relative_path):
    """ 获取资源文件的绝对路径，适配 PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==========================================
# 1. 头部姿态估算模块 (Head Pose Estimation)
# ==========================================
class HeadPoseEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def get_pose(self, image):
        """
        输入图像，返回头部姿态 (pitch, yaw, roll)、下巴坐标 和 绘制了信息的图像
        """
        start = time.time()
        
        # 转换颜色空间 BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        
        pitch, yaw, roll = 0, 0, 0
        face_detected = False
        chin_y = img_h  # 默认下巴在最底部

        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                # 获取关键点 (MediaPipe 468点模型)
                # 33: 左眼角, 263: 右眼角, 1: 鼻尖, 61: 左嘴角, 291: 右嘴角, 199: 下巴, 152: 下巴底端
                key_landmarks = [33, 263, 1, 61, 291, 199]
                
                # 获取下巴底端坐标 (索引 152) 用于手部判定
                chin_landmark = face_landmarks.landmark[152]
                chin_y = int(chin_landmark.y * img_h)

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in key_landmarks or idx == 1: # 1 is nose tip
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    
                    # 获取 2D 和 3D 坐标
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])       

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # 摄像机矩阵 (假设)
                focal_length = 1 * img_w
                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # 畸变系数 (假设无畸变)
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # 获取旋转矩阵
                rmat, jac = cv2.Rodrigues(rot_vec)

                # 获取欧拉角
                try:
                    # 尝试 7 个返回值
                    angles, mtxR, mtxQ, Q, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                except ValueError:
                    try:
                        # 尝试 6 个返回值
                        ret_val = cv2.RQDecomp3x3(rmat)
                        angles = ret_val[0]
                    except Exception as e:
                        print(f"RQDecomp3x3 Error: {e}")
                        pitch, yaw, roll = 0, 0, 0
                        continue

                # 转换角度
                pitch = angles[0] * 360
                yaw = angles[1] * 360
                roll = angles[2] * 360 

                # 绘制轴线 (可视化)
                nose_3d_projection, jac = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + yaw * 10), int(nose_2d[1] - pitch * 10))
                
                cv2.line(image, p1, p2, (255, 0, 0), 3)

                cv2.putText(image, f"Pitch: {int(pitch)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(image, f"Yaw: {int(yaw)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 绘制下巴线辅助线 (调试用)
                cv2.line(image, (0, chin_y), (img_w, chin_y), (100, 100, 100), 1)

                break

        return image, pitch, yaw, roll, face_detected, chin_y

# ==========================================
# 1.5 物体检测模块 (Object Detection - 识别手机)
# ==========================================
class ObjectDetector:
    def __init__(self, model_filename='efficientdet_lite0.tflite'):
        self.mp_tasks = mp.tasks
        self.mp_vision = mp.tasks.vision
        
        # 获取模型文件的绝对路径 (适配打包环境)
        model_path = resource_path(model_filename)
        
        # 加载 TFLite 模型
        try:
            base_options = self.mp_tasks.BaseOptions(model_asset_path=model_path)
            # 使用 Video 模式提高稳定性
            options = self.mp_vision.ObjectDetectorOptions(
                base_options=base_options,
                running_mode=self.mp_vision.RunningMode.VIDEO,
                score_threshold=0.4, # 提高阈值减少误报
                category_allowlist=['cell phone'] # 只检测手机
            )
            self.detector = self.mp_vision.ObjectDetector.create_from_options(options)
            self.last_timestamp_ms = 0
            self.is_loaded = True
        except Exception as e:
            print(f"Object Detector load failed: {e}")
            self.is_loaded = False

    def check_phone(self, image):
        """
        检测画面中是否有手机
        """
        if not self.is_loaded:
            return image, False

        # MediaPipe 需要 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # 计算时间戳 (MediaPipe Video 模式需要递增的时间戳)
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        phone_detected = False
        img_h, img_w, _ = image.shape

        for detection in detection_result.detections:
            # 绘制框
            bbox = detection.bounding_box
            start_point = (int(bbox.origin_x), int(bbox.origin_y))
            end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
            
            # 绘制红色矩形框
            cv2.rectangle(image, start_point, end_point, (0, 0, 255), 3)
            
            category = detection.categories[0]
            label = f"{category.category_name} ({category.score:.2f})"
            cv2.putText(image, label, (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if category.category_name == 'cell phone':
                phone_detected = True

        return image, phone_detected
    
# ==========================================
# 1.6 手部检测模块 (Hand Detection)
# ==========================================
class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def check_hands(self, image, chin_y):
        """
        检测手部是否抬起超过下巴位置 (判定为玩手机/打电话/吃东西)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hands_raised = False
        img_h, img_w, _ = image.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部骨架
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 检查关键点高度
                # 检查指尖 (8, 12, 16, 20) 和 掌心 (9)
                # 注意：y 坐标越小越靠上
                for idx in [8, 12, 16, 20, 9]:
                    y = int(hand_landmarks.landmark[idx].y * img_h)
                    # 如果手部关键点高于下巴 (y < chin_y)
                    # 且不是特别靠边缘 (排除打字时的手部边缘误触)
                    if y < chin_y - 20: 
                        hands_raised = True
                        break
        
        return image, hands_raised

# ==========================================
# 2. 音频警报模块 (Audio Alerter - 语音版)
# ==========================================
class AudioAlerter:
    def __init__(self):
        self.is_playing = False
        self.stop_event = threading.Event()
        self.thread = None

    def play_alert(self):
        if not self.is_playing:
            self.is_playing = True
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._speech_loop)
            self.thread.daemon = True
            self.thread.start()

    def _speech_loop(self):
        """循环播放语音，直到收到停止信号"""
        while not self.stop_event.is_set():
            # 使用 macOS 的 'say' 命令
            # "Ting-Ting" 是中文语音包，如果没有会回退到默认
            # 如果觉得语速太慢，可以加 -r 参数调整语速
            try:
                subprocess.run(["say", "别摸鱼了，别摸鱼了"], check=False)
            except Exception as e:
                print(f"Speech error: {e}")
                
            # 简单的防刷屏延迟，防止语速过快听不清
            time.sleep(0.5) 

    def stop_alert(self):
        if self.is_playing:
            self.stop_event.set()
            self.is_playing = False
            # 这里的线程会因为 say 命令结束而自然退出

    def cleanup(self):
        self.stop_alert()


# ==========================================
# 3. 主应用程序 (Attention App)
# ==========================================
class AttentionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("注意力监控工具 (Attention Monitor)")
        self.root.geometry("900x700")
        
        # --- 变量初始化 ---
        self.is_running = False
        self.cap = None
        _log("AttentionApp: 创建 HeadPoseEstimator")
        self.estimator = HeadPoseEstimator()
        _log("AttentionApp: 创建 HandDetector")
        self.hand_detector = HandDetector()
        _log("AttentionApp: 创建 ObjectDetector")
        self.object_detector = ObjectDetector() # 初始化手机检测
        _log("AttentionApp: 创建 AudioAlerter")
        self.alerter = AudioAlerter()
        _log("AttentionApp: 创建 UI")
        
        # 状态变量
        self.start_distraction_time = None
        self.distraction_duration = 0
        self.status_text = "待机 (Standby)"
        self.status_color = "gray"
        
        # 阈值设置 (默认值)
        self.YAW_THRESHOLD = 20
        self.PITCH_THRESHOLD = 15
        self.PHONE_PITCH_THRESHOLD = 25 # 玩手机通常头会更低
        self.TIME_THRESHOLD = 3.0 # 秒
        
        # --- UI 布局 ---
        self._setup_ui()
        _log("AttentionApp: _setup_ui 完成")

    def _setup_ui(self):
        # 1. 视频显示区域
        self.video_frame = tk.Frame(self.root, bg="black")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 2. 控制面板
        control_panel = tk.Frame(self.root, pady=10)
        control_panel.pack(fill=tk.X, side=tk.BOTTOM)
        
        # 按钮
        btn_frame = tk.Frame(control_panel)
        btn_frame.pack(side=tk.LEFT, padx=20)
        
        self.btn_start = ttk.Button(btn_frame, text="开始监控 (Start)", command=self.start_monitoring)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(btn_frame, text="停止监控 (Stop)", command=self.stop_monitoring, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        # 灵敏度设置
        slider_frame = tk.Frame(control_panel)
        slider_frame.pack(side=tk.RIGHT, padx=20)
        
        tk.Label(slider_frame, text="容忍时间 (秒):").pack(side=tk.LEFT)
        self.lbl_threshold = tk.Label(slider_frame, text=f"{self.TIME_THRESHOLD}s") # 先创建 label
        self.time_slider = ttk.Scale(slider_frame, from_=0.5, to=10.0, orient=tk.HORIZONTAL, length=150, command=self._update_threshold)
        self.time_slider.set(self.TIME_THRESHOLD)
        self.time_slider.pack(side=tk.LEFT, padx=5)
        self.lbl_threshold.pack(side=tk.LEFT) # 后 pack

        # 状态栏
        self.lbl_status = tk.Label(self.root, text=self.status_text, font=("Arial", 14, "bold"), fg=self.status_color, bg="#f0f0f0", pady=5)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

    def _update_threshold(self, val):
        self.TIME_THRESHOLD = round(float(val), 1)
        self.lbl_threshold.config(text=f"{self.TIME_THRESHOLD}s")

    def start_monitoring(self):
        if not self.is_running:
            print("Attempting to open camera with AVFoundation...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            
            if not self.cap.isOpened():
                print("AVFoundation failed. Trying default backend...")
                self.cap = cv2.VideoCapture(0)
                
            if not self.cap.isOpened():
                print("Default backend failed. Trying index 1 with AVFoundation...")
                self.cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
            
            if not self.cap.isOpened():
                print("All attempts failed.")
                tk.messagebox.showerror("错误", "无法打开摄像头！")
                return
            
            # 关键修改：增加短暂延迟让摄像头预热
            time.sleep(1.0)
            
            # 检查第一帧
            print("Reading initial frame...")
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Initial frame empty. Trying to re-read multiple times...")
                # 尝试重试读取几次，有时候摄像头启动需要时间
                for i in range(5):
                    print(f"Retry {i+1}...")
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print("Frame captured successfully!")
                        break
                    time.sleep(0.5)
            
            if not ret or frame is None:
                print("Still failed to capture frame. Aborting.")
                self.cap.release()
                tk.messagebox.showerror("错误", "摄像头已打开但无法获取画面（黑屏）")
                return

            print("Camera started successfully.")
            self.is_running = True
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.start_distraction_time = None
            self._process_frame()

    def stop_monitoring(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.alerter.stop_alert()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.canvas.delete("all")
        self.lbl_status.config(text="待机 (Standby)", fg="gray")

    def _process_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_monitoring()
            return

        # 镜像翻转，像照镜子一样
        frame = cv2.flip(frame, 1)
        
        # 姿态估计
        frame, pitch, yaw, roll, face_detected, chin_y = self.estimator.get_pose(frame)
        
        # 手部检测
        frame, hands_raised = self.hand_detector.check_hands(frame, chin_y)
        
        # 手机检测 (Object Detection)
        frame, phone_detected = self.object_detector.check_phone(frame)

        # --- 摸鱼判定逻辑 ---
        is_distracted = False
        status_msg = "专注 (FOCUSED)"
        color = (0, 255, 0) # Green

        if not face_detected:
            is_distracted = True
            status_msg = "未检测到人脸 (NO FACE)"
            color = (0, 165, 255) # Orange
        else:
            # 0. 优先级最高：直接检测到手机
            if phone_detected:
                is_distracted = True
                status_msg = "检测到手机! (PHONE)"
                color = (0, 0, 255) # Red

            # 1. 检查是否低头太严重 (玩手机/睡觉)
            elif pitch > self.PHONE_PITCH_THRESHOLD: # Pitch向下为正 (通常)
                is_distracted = True
                status_msg = "低头玩手机 (HEAD DOWN)"
                color = (0, 0, 255)
            
            # 2. 检查手部是否抬起 (打电话/吃东西)
            elif hands_raised:
                is_distracted = True
                status_msg = "手部干扰 (HANDS UP)"
                color = (255, 0, 255) # Magenta

            # 3. 检查常规视线偏离
            elif abs(pitch) > self.PITCH_THRESHOLD or abs(yaw) > self.YAW_THRESHOLD:
                is_distracted = True
                status_msg = "视线偏离 (LOOKING AWAY)"
                color = (0, 255, 255) # Yellow

        # 计时器逻辑
        current_time = time.time()
        
        if is_distracted:
            if self.start_distraction_time is None:
                self.start_distraction_time = current_time
            
            elapsed = current_time - self.start_distraction_time
            
            # 显示倒计时进度条或文字
            cv2.putText(frame, f"Time: {elapsed:.1f}s / {self.TIME_THRESHOLD}s", (20, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if elapsed > self.TIME_THRESHOLD:
                # 触发警报
                status_msg = "警报：摸鱼中！(ALERT)"
                color = (0, 0, 255) # Red
                cv2.putText(frame, "ALERT: FOCUS!", (int(frame.shape[1]/2)-150, int(frame.shape[0]/2)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                self.alerter.play_alert()
            else:
                self.alerter.stop_alert()
        else:
            # 重置计时器
            self.start_distraction_time = None
            self.alerter.stop_alert()

        # 更新 UI 状态文字
        self.lbl_status.config(text=status_msg, fg=self._bgr_to_hex(color))

        # 转换图像格式以显示在 Tkinter
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # 调整大小以适应 Canvas (保持比例)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            img_pil = self._resize_image(img_pil, canvas_width, canvas_height)
        
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # 清除上一帧 (防止内存泄漏)
        self.canvas.delete("all")
        
        # 居中显示
        x_center = canvas_width // 2
        y_center = canvas_height // 2
        self.canvas.create_image(x_center, y_center, anchor=tk.CENTER, image=img_tk)
        
        self.canvas.image = img_tk # 防止垃圾回收

        # 循环调用
        self.root.after(10, self._process_frame)

    def _resize_image(self, image, max_width, max_height):
        # 保持纵横比缩放
        width, height = image.size
        ratio = min(max_width/width, max_height/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _bgr_to_hex(self, bgr):
        # OpenCV 使用 BGR，Tkinter 使用 Hex RGB
        return "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])

    def on_closing(self):
        self.stop_monitoring()
        self.alerter.cleanup()
        self.root.destroy()

# ==========================================
# 4. 程序入口
# ==========================================
if __name__ == "__main__":
    try:
        _log("main: 即将创建 Tk 窗口")
        root = tk.Tk()
        _log("main: Tk 已创建，创建 AttentionApp")
        app = AttentionApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        _log("main: 进入 mainloop")
        root.mainloop()
        _log("main: mainloop 已退出（正常关闭）")
    except Exception as e:
        error_msg = traceback.format_exc()
        _log("main: 捕获到异常:\n" + error_msg)
        try:
            with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write("\n--- 完整 traceback ---\n" + error_msg + "\n")
        except Exception:
            pass
        show_critical_error("Runtime Error:\n" + error_msg[:500])
        sys.exit(1)
