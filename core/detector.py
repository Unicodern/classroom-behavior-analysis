"""
行为检测模块
基于YOLOv8-pose官方预训练模型实现课堂行为检测
支持行为：举手、低头、趴桌、坐姿、讨论
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Union, Generator, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Keypoint:
    """人体关键点数据类"""
    x: float           # x坐标
    y: float           # y坐标
    confidence: float  # 置信度
    
    def is_valid(self, threshold: float = 0.5) -> bool:
        """检查关键点是否有效（置信度达标）"""
        return self.confidence >= threshold


@dataclass
class Person:
    """单个人体检测结果"""
    person_id: int                    # 人员编号
    keypoints: List[Keypoint]         # 17个关键点
    bbox: Tuple[float, float, float, float]  # 边界框 (x1, y1, x2, y2)
    confidence: float                 # 整体置信度
    
    # COCO格式17个关键点索引定义
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def get_keypoint(self, name: str) -> Optional[Keypoint]:
        """根据名称获取关键点"""
        if name in self.KEYPOINT_NAMES:
            idx = self.KEYPOINT_NAMES.index(name)
            return self.keypoints[idx] if idx < len(self.keypoints) else None
        return None


@dataclass
class BehaviorResult:
    """行为检测结果"""
    timestamp: datetime      # 时间戳
    person_id: int           # 人员编号
    behavior: str            # 行为类型
    confidence: float        # 置信度
    keypoints_data: Dict     # 关键点原始数据


class BehaviorDetector:
    """
    课堂行为检测器
    基于YOLOv8-pose官方预训练模型，无需训练即可使用
    """
    
    # 行为类型定义
    BEHAVIORS = {
        'raise_hand': '举手',
        'head_down': '低头',
        'lying_on_desk': '趴桌',
        'good_posture': '坐姿端正',
        'discussing': '讨论中',
        'unknown': '未知'
    }
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.5, 
                 device: str = 'auto'):
        """
        初始化行为检测器
        
        Args:
            model_path: YOLOv8模型路径，默认使用models/yolov8n-pose.pt
            conf_threshold: 检测置信度阈值
            device: 运行设备 ('cpu', 'cuda', 'auto')
        """
        self.conf_threshold = conf_threshold
        
        # 设置模型路径
        if model_path is None:
            model_path = Path(__file__).parent.parent / 'models' / 'yolov8n-pose.pt'
        self.model_path = str(model_path)
        
        # 延迟导入ultralytics，避免未安装时出错
        try:
            from ultralytics import YOLO
            from torch.serialization import add_safe_globals
        except ImportError:
            raise ImportError("请先安装ultralytics: pip install ultralytics>=8.0.200")
        
        # 加载YOLOv8-pose模型
        logger.info(f"正在加载YOLOv8模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # 设置运行设备
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        logger.info(f"使用设备: {self.device}")
        
        # 行为统计
        self.frame_count = 0
        self.detected_persons = 0
        
    def detect(self, frame: np.ndarray) -> List[Person]:
        """
        对单帧图像进行人体姿态检测
        
        Args:
            frame: BGR格式的图像数组 (H, W, 3)
            
        Returns:
            List[Person]: 检测到的人体列表
        """
        # YOLOv8推理
        results = self.model(frame, device=self.device, verbose=False)
        
        persons = []
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints
            boxes_data = results[0].boxes
            
            # 解析每个人体的检测结果
            for i, (kpts, box) in enumerate(zip(keypoints_data, boxes_data)):
                # 提取关键点
                keypoints = []
                kpts_array = kpts.xy.cpu().numpy()[0] if hasattr(kpts, 'xy') else kpts.data.cpu().numpy()[0]
                conf_array = kpts.conf.cpu().numpy()[0] if hasattr(kpts, 'conf') else [1.0] * 17
                
                for j in range(min(17, len(kpts_array))):
                    x, y = kpts_array[j]
                    conf = conf_array[j] if j < len(conf_array) else 1.0
                    keypoints.append(Keypoint(float(x), float(y), float(conf)))
                
                # 提取边界框
                if hasattr(box, 'xyxy'):
                    bbox = tuple(box.xyxy.cpu().numpy()[0])
                else:
                    bbox = (0, 0, frame.shape[1], frame.shape[0])
                
                # 提取置信度
                box_conf = float(box.conf.cpu().numpy()[0]) if hasattr(box, 'conf') else 1.0
                
                person = Person(
                    person_id=i,
                    keypoints=keypoints,
                    bbox=bbox,
                    confidence=box_conf
                )
                persons.append(person)
        
        self.detected_persons = len(persons)
        return persons
    
    def classify_behavior(self, person: Person) -> Tuple[str, float]:
        """
        根据关键点分类行为
        
        检测逻辑（按优先级排序）：
        1. 举手：手腕明显高于肩膀（高优先级）
        2. 趴桌：头部远低于肩膀（趴在桌上）
        3. 低头：头部明显低于正常位置
        4. 坐姿端正：脊柱角度正常
        5. 讨论：需要多人互动检测（在更高层处理）
        
        Args:
            person: 人体检测结果
            
        Returns:
            Tuple[str, float]: (行为类型, 置信度)
        """
        # 获取关键关键点
        nose = person.get_keypoint('nose')
        left_shoulder = person.get_keypoint('left_shoulder')
        right_shoulder = person.get_keypoint('right_shoulder')
        left_elbow = person.get_keypoint('left_elbow')
        right_elbow = person.get_keypoint('right_elbow')
        left_wrist = person.get_keypoint('left_wrist')
        right_wrist = person.get_keypoint('right_wrist')
        left_hip = person.get_keypoint('left_hip')
        right_hip = person.get_keypoint('right_hip')
        left_ear = person.get_keypoint('left_ear')
        right_ear = person.get_keypoint('right_ear')
        
        # 计算参考基准（肩部和臀部的平均高度）
        shoulder_y = None
        if left_shoulder and right_shoulder:
            if left_shoulder.is_valid() and right_shoulder.is_valid():
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        
        hip_y = None
        if left_hip and right_hip:
            if left_hip.is_valid() and right_hip.is_valid():
                hip_y = (left_hip.y + right_hip.y) / 2
        
        # 获取人体高度作为参考（用于自适应阈值）
        person_height = self._estimate_person_height(person)
        
        # 1. 检测举手行为（最高优先级）
        raise_hand_conf = self._check_raise_hand_v2(
            left_wrist, right_wrist, left_shoulder, right_shoulder, shoulder_y
        )
        if raise_hand_conf > 0.6:
            return 'raise_hand', raise_hand_conf
        
        # 2. 检测趴桌行为（头部远低于肩膀）
        lying_conf = self._check_lying_on_desk_v2(nose, shoulder_y, person_height)
        if lying_conf > 0.6:
            return 'lying_on_desk', lying_conf
        
        # 3. 检测低头行为（头部低于正常位置）
        head_down_conf = self._check_head_down_v2(nose, left_ear, right_ear, shoulder_y, person_height)
        if head_down_conf > 0.6:
            return 'head_down', head_down_conf
        
        # 4. 检测坐姿端正
        posture_conf = self._check_good_posture_v2(nose, shoulder_y, hip_y, person_height)
        if posture_conf > 0.5:
            return 'good_posture', posture_conf
        
        # 如果都不符合，返回未知但给出最高置信度的行为
        best_behavior, best_conf = self._get_best_behavior(
            raise_hand_conf, lying_conf, head_down_conf, posture_conf
        )
        return best_behavior, best_conf
    
    def _estimate_person_height(self, person: Person) -> float:
        """估计人体高度（用于自适应阈值）"""
        # 获取有效关键点的y坐标范围
        valid_y = [kp.y for kp in person.keypoints if kp.is_valid(0.3)]
        if len(valid_y) >= 5:  # 至少5个有效关键点
            return max(valid_y) - min(valid_y)
        return 200  # 默认高度200像素
    
    def _check_raise_hand_v2(self, left_wrist, right_wrist, 
                             left_shoulder, right_shoulder, shoulder_y) -> float:
        """
        检测举手行为（返回置信度0-1）
        手腕明显高于肩膀
        """
        if shoulder_y is None:
            return 0.0
        
        max_conf = 0.0
        
        # 检查左手
        if left_wrist and left_shoulder:
            if left_wrist.is_valid(0.3) and left_shoulder.is_valid(0.3):
                # 手腕高于肩膀，差值越大置信度越高
                diff = left_shoulder.y - left_wrist.y
                if diff > 10:  # 至少高10像素
                    conf = min(0.95, 0.6 + diff / 100)  # 线性增长，最高0.95
                    max_conf = max(max_conf, conf)
        
        # 检查右手
        if right_wrist and right_shoulder:
            if right_wrist.is_valid(0.3) and right_shoulder.is_valid(0.3):
                diff = right_shoulder.y - right_wrist.y
                if diff > 10:
                    conf = min(0.95, 0.6 + diff / 100)
                    max_conf = max(max_conf, conf)
        
        return max_conf
    
    def _check_lying_on_desk_v2(self, nose, shoulder_y, person_height) -> float:
        """
        检测趴桌行为（返回置信度0-1）
        头部远低于肩膀（趴在桌上）
        """
        if nose is None or shoulder_y is None:
            return 0.0
        
        if nose.is_valid(0.3):
            # 鼻子位置低于肩膀
            diff = nose.y - shoulder_y
            # 自适应阈值：人体高度的40%
            threshold = person_height * 0.4
            
            if diff > threshold:
                # 差值越大置信度越高
                conf = min(0.95, 0.6 + (diff - threshold) / 100)
                return conf
        
        return 0.0
    
    def _check_head_down_v2(self, nose, left_ear, right_ear, shoulder_y, person_height) -> float:
        """
        检测低头行为（返回置信度0-1）
        头部低于正常位置，但还没到趴桌的程度
        """
        if nose is None or shoulder_y is None:
            return 0.0
        
        conf = 0.0
        
        # 方法1：通过鼻子位置判断
        if nose.is_valid(0.3):
            diff = nose.y - shoulder_y
            # 自适应阈值：人体高度的15%-40%
            low_threshold = person_height * 0.15
            high_threshold = person_height * 0.4
            
            if low_threshold < diff < high_threshold:
                # 在阈值范围内，越接近中间置信度越高
                mid = (low_threshold + high_threshold) / 2
                distance_from_mid = abs(diff - mid)
                conf = 0.95 - (distance_from_mid / (high_threshold - low_threshold)) * 0.3
                conf = max(0.6, min(0.95, conf))
        
        # 方法2：通过耳朵位置辅助判断（如果耳朵可见且位置偏低）
        ear_y = None
        if left_ear and left_ear.is_valid(0.3):
            ear_y = left_ear.y
        elif right_ear and right_ear.is_valid(0.3):
            ear_y = right_ear.y
        
        if ear_y and shoulder_y:
            if ear_y > shoulder_y:  # 耳朵低于肩膀，说明头低下
                ear_conf = min(0.9, 0.6 + (ear_y - shoulder_y) / 80)
                conf = max(conf, ear_conf)
        
        return conf
    
    def _check_good_posture_v2(self, nose, shoulder_y, hip_y, person_height) -> float:
        """
        检测坐姿端正（返回置信度0-1）
        脊柱角度正常，头部在肩膀上方或略低
        """
        if nose is None or shoulder_y is None:
            return 0.0
        
        if nose.is_valid(0.3):
            diff = nose.y - shoulder_y
            # 自适应阈值：人体高度的15%以内算正常
            threshold = person_height * 0.15
            
            if diff < threshold:
                # 差值越小置信度越高
                conf = 0.95 - (diff / threshold) * 0.3
                return max(0.5, min(0.95, conf))
        
        return 0.0
    
    def _get_best_behavior(self, raise_hand_conf, lying_conf, 
                           head_down_conf, posture_conf) -> Tuple[str, float]:
        """返回置信度最高的行为"""
        behaviors = [
            ('raise_hand', raise_hand_conf),
            ('lying_on_desk', lying_conf),
            ('head_down', head_down_conf),
            ('good_posture', posture_conf)
        ]
        
        # 按置信度排序
        behaviors.sort(key=lambda x: x[1], reverse=True)
        best_behavior, best_conf = behaviors[0]
        
        # 如果最高置信度太低，返回unknown
        if best_conf < 0.3:
            return 'unknown', best_conf
        
        return best_behavior, best_conf
    
    def detect_behaviors(self, frame: np.ndarray) -> List[BehaviorResult]:
        """
        检测帧中所有人员的行为
        
        Args:
            frame: 输入图像帧
            
        Returns:
            List[BehaviorResult]: 行为检测结果列表
        """
        timestamp = datetime.now()
        persons = self.detect(frame)
        results = []
        
        for person in persons:
            behavior, confidence = self.classify_behavior(person)
            
            # 提取关键点数据用于记录
            keypoints_data = {
                name: {
                    'x': kp.x,
                    'y': kp.y,
                    'conf': kp.confidence
                }
                for name, kp in zip(Person.KEYPOINT_NAMES, person.keypoints)
                if kp.is_valid(self.conf_threshold)
            }
            
            result = BehaviorResult(
                timestamp=timestamp,
                person_id=person.person_id,
                behavior=behavior,
                confidence=confidence,
                keypoints_data=keypoints_data
            )
            results.append(result)
        
        return results
    
    def process_video(self, source: Union[str, int], 
                      display: bool = True,
                      save_path: str = None) -> Generator[Dict, None, None]:
        """
        处理视频流（摄像头或视频文件）
        
        Args:
            source: 视频源（摄像头索引或视频文件路径）
            display: 是否实时显示
            save_path: 保存处理结果的路径
            
        Yields:
            Dict: 每帧的检测结果
        """
        # 打开视频源
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        # 判断是否为摄像头
        is_camera = isinstance(source, int)
        
        # 使用默认后端打开摄像头（避免DSHOW兼容性问题）
        cap = cv2.VideoCapture(source)
        
        # 如果默认后端失败且是摄像头，尝试DSHOW
        if not cap.isOpened() and is_camera:
            logger.warning(f"默认后端无法打开摄像头，尝试DSHOW后端...")
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频源: {source}")
        
        # 获取视频信息（使用摄像头实际支持的分辨率）
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        source_type = "摄像头" if is_camera else "视频文件"
        logger.info(f"{source_type}信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")
        
        # 摄像头预热（读取几帧丢弃，确保摄像头准备好）
        if is_camera:
            logger.info("正在预热摄像头...")
            for _ in range(5):
                ret, _ = cap.read()
                if not ret:
                    break
            logger.info("摄像头预热完成")
        
        # 初始化视频写入器
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height), isColor=True)
            logger.info(f"视频将保存至: {save_path}")
        
        self.frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("视频读取结束")
                    break
                
                self.frame_count += 1
                
                # 行为检测
                persons = self.detect(frame)
                behaviors = []
                
                # 可视化
                vis_frame = frame.copy()
                for person in persons:
                    behavior, conf = self.classify_behavior(person)
                    behaviors.append({
                        'person_id': person.person_id,
                        'behavior': behavior,
                        'confidence': conf
                    })
                    
                    # 绘制边界框和行为标签
                    vis_frame = self._draw_detection(vis_frame, person, behavior, conf)
                
                # 添加帧信息
                info_text = f"Frame: {self.frame_count} | Persons: {len(persons)}"
                cv2.putText(vis_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 保存视频
                if writer:
                    writer.write(vis_frame)
                
                # 显示
                if display:
                    cv2.imshow('Behavior Detection', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("用户中断检测")
                        break
                
                # 返回检测结果
                yield {
                    'frame_id': self.frame_count,
                    'timestamp': datetime.now(),
                    'person_count': len(persons),
                    'behaviors': behaviors,
                    'frame': vis_frame
                }
                
        finally:
            # 释放摄像头资源
            logger.info("正在释放视频资源...")
            cap.release()
            if writer:
                writer.release()
            if display:
                # 延迟一点再关闭窗口，确保所有帧都显示完成
                cv2.waitKey(100)
                cv2.destroyAllWindows()
                cv2.waitKey(100)
            logger.info(f"处理完成，共处理 {self.frame_count} 帧")
    
    def _draw_detection(self, frame: np.ndarray, person: Person, 
                        behavior: str, confidence: float) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            frame: 原始图像
            person: 人体检测结果
            behavior: 行为类型
            confidence: 置信度
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        result = frame.copy()
        
        # 绘制边界框
        x1, y1, x2, y2 = map(int, person.bbox)
        
        # 根据行为类型设置颜色
        color_map = {
            'raise_hand': (0, 255, 0),      # 绿色-举手
            'head_down': (0, 165, 255),     # 橙色-低头
            'lying_on_desk': (0, 0, 255),   # 红色-趴桌
            'good_posture': (255, 255, 0),  # 青色-坐姿端正
            'discussing': (255, 0, 255),    # 紫色-讨论
            'unknown': (128, 128, 128)      # 灰色-未知
        }
        color = color_map.get(behavior, (128, 128, 128))
        
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # 绘制行为标签（使用PIL支持中文）
        label = f"{self.BEHAVIORS.get(behavior, behavior)}: {confidence:.2f}"
        result = self._draw_chinese_text(result, label, (x1, y1 - 10), color, 20)
        
        # 绘制关键点
        for i, kp in enumerate(person.keypoints):
            if kp.is_valid(self.conf_threshold):
                cv2.circle(result, (int(kp.x), int(kp.y)), 3, (0, 255, 255), -1)
        
        # 绘制骨架连线
        skeleton = [
            [5, 7], [7, 9],     # 左臂
            [6, 8], [8, 10],    # 右臂
            [5, 6],             # 肩膀
            [5, 11], [6, 12],   # 躯干
            [11, 13], [13, 15], # 左腿
            [12, 14], [14, 16], # 右腿
            [11, 12]            # 臀部
        ]
        
        for connection in skeleton:
            kp1 = person.keypoints[connection[0]]
            kp2 = person.keypoints[connection[1]]
            if kp1.is_valid(self.conf_threshold) and kp2.is_valid(self.conf_threshold):
                pt1 = (int(kp1.x), int(kp1.y))
                pt2 = (int(kp2.x), int(kp2.y))
                cv2.line(result, pt1, pt2, (255, 255, 255), 1)
        
        return result
    
    def _draw_chinese_text(self, frame: np.ndarray, text: str, 
                           position: Tuple[int, int], 
                           bg_color: Tuple[int, int, int], 
                           font_size: int = 20) -> np.ndarray:
        """
        使用PIL绘制中文文本
        
        Args:
            frame: OpenCV图像(BGR格式)
            text: 要绘制的中文文本
            position: 文本位置(x, y)
            bg_color: 背景颜色(BGR格式)
            font_size: 字体大小
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        # 转换BGR到RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # 尝试加载中文字体
        font = None
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑粗体
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        
        if font is None:
            # 使用默认字体
            font = ImageFont.load_default()
        
        # 获取文本尺寸
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x, y = position
        # 确保文本在图像内
        y = max(y, text_height + 5)
        
        # 绘制背景矩形
        padding = 3
        draw.rectangle(
            [x, y - text_height - padding, x + text_width, y + padding],
            fill=(bg_color[2], bg_color[1], bg_color[0])  # BGR转RGB
        )
        
        # 绘制文本（白色）
        draw.text((x, y - text_height), text, font=font, fill=(255, 255, 255))
        
        # 转换回BGR
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return result
    
    def get_statistics(self) -> Dict:
        """获取检测统计信息"""
        return {
            'total_frames': self.frame_count,
            'total_persons_detected': self.detected_persons
        }
