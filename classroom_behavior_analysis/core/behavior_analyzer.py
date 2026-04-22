"""
行为统计分析模块
用于统计和分析课堂行为数据，计算专注度等指标
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class BehaviorEvent:
    """单个行为事件记录"""
    timestamp: datetime
    person_id: int
    behavior: str
    confidence: float
    duration: float = 0.0  # 行为持续时长（秒）


@dataclass
class TimeSegment:
    """时间段统计"""
    start_time: datetime
    end_time: datetime
    behavior_counts: Dict[str, int] = field(default_factory=dict)
    person_count: int = 0
    focus_rate: float = 0.0


class BehaviorAnalyzer:
    """
    课堂行为分析器
    统计行为频次、计算专注度、生成分析报告
    """
    
    # 行为权重定义（用于计算专注度）
    BEHAVIOR_WEIGHTS = {
        'raise_hand': 1.0,      # 举手 - 积极行为
        'good_posture': 0.8,    # 坐姿端正 - 正面行为
        'discussing': 0.6,      # 讨论中 - 中性偏积极
        'head_down': 0.3,       # 低头 - 负面行为
        'lying_on_desk': 0.0,   # 趴桌 - 严重负面行为
        'unknown': 0.5          # 未知 - 中性
    }
    
    def __init__(self, segment_duration: int = 60):
        """
        初始化行为分析器
        
        Args:
            segment_duration: 时间段划分间隔（秒），默认60秒
        """
        self.segment_duration = segment_duration
        
        # 原始事件记录
        self.events: List[BehaviorEvent] = []
        
        # 时间段统计
        self.time_segments: List[TimeSegment] = []
        
        # 总体统计
        self.total_counts: Dict[str, int] = defaultdict(int)
        self.person_behavior_history: Dict[int, List[BehaviorEvent]] = defaultdict(list)
        
        # 时间范围
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # 学生人数
        self.student_count: int = 0
        
        logger.info(f"行为分析器初始化完成，时间段间隔: {segment_duration}秒")
    
    def update(self, behaviors: List[Dict], timestamp: Optional[datetime] = None):
        """
        更新行为统计数据
        
        Args:
            behaviors: 行为检测结果列表
            timestamp: 时间戳，默认使用当前时间
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 更新时间范围
        if self.start_time is None:
            self.start_time = timestamp
        self.end_time = timestamp
        
        # 记录事件
        for behavior_data in behaviors:
            event = BehaviorEvent(
                timestamp=timestamp,
                person_id=behavior_data.get('person_id', 0),
                behavior=behavior_data.get('behavior', 'unknown'),
                confidence=behavior_data.get('confidence', 0.5)
            )
            self.events.append(event)
            
            # 更新统计
            self.total_counts[event.behavior] += 1
            self.person_behavior_history[event.person_id].append(event)
        
        # 更新学生人数
        if behaviors:
            max_id = max(b['person_id'] for b in behaviors)
            self.student_count = max(self.student_count, max_id + 1)
    
    def calculate_focus_rate(self, behaviors: List[str]) -> float:
        """
        计算专注度指数
        
        Args:
            behaviors: 行为类型列表
            
        Returns:
            float: 专注度指数 (0-1)
        """
        if not behaviors:
            return 0.0
        
        total_weight = sum(self.BEHAVIOR_WEIGHTS.get(b, 0.5) for b in behaviors)
        return total_weight / len(behaviors)
    
    def analyze_time_segment(self, start: datetime, end: datetime) -> TimeSegment:
        """
        分析指定时间段的行为数据
        
        Args:
            start: 开始时间
            end: 结束时间
            
        Returns:
            TimeSegment: 时间段统计结果
        """
        # 筛选时间段内的事件
        segment_events = [
            e for e in self.events 
            if start <= e.timestamp <= end
        ]
        
        # 统计行为频次
        behavior_counts = defaultdict(int)
        for event in segment_events:
            behavior_counts[event.behavior] += 1
        
        # 统计人数
        person_ids = set(e.person_id for e in segment_events)
        person_count = len(person_ids)
        
        # 计算专注度
        behaviors = [e.behavior for e in segment_events]
        focus_rate = self.calculate_focus_rate(behaviors)
        
        return TimeSegment(
            start_time=start,
            end_time=end,
            behavior_counts=dict(behavior_counts),
            person_count=person_count,
            focus_rate=focus_rate
        )
    
    def generate_time_segments(self) -> List[TimeSegment]:
        """
        生成所有时间段的统计
        
        Returns:
            List[TimeSegment]: 时间段统计列表
        """
        if self.start_time is None or self.end_time is None:
            return []
        
        self.time_segments = []
        current = self.start_time
        
        while current < self.end_time:
            segment_end = min(
                current + timedelta(seconds=self.segment_duration),
                self.end_time
            )
            
            segment = self.analyze_time_segment(current, segment_end)
            self.time_segments.append(segment)
            
            current = segment_end
        
        return self.time_segments
    
    def get_behavior_distribution(self) -> Dict[str, Dict]:
        """
        获取行为分布统计
        
        Returns:
            Dict: 各行为的详细统计
        """
        total = sum(self.total_counts.values())
        if total == 0:
            return {}
        
        distribution = {}
        for behavior, count in self.total_counts.items():
            distribution[behavior] = {
                'count': count,
                'percentage': round(count / total * 100, 2),
                'weight': self.BEHAVIOR_WEIGHTS.get(behavior, 0.5)
            }
        
        return distribution
    
    def get_person_behavior_summary(self, person_id: int) -> Dict:
        """
        获取指定人员的行为摘要
        
        Args:
            person_id: 人员编号
            
        Returns:
            Dict: 行为摘要统计
        """
        events = self.person_behavior_history.get(person_id, [])
        if not events:
            return {}
        
        behavior_counts = defaultdict(int)
        for event in events:
            behavior_counts[event.behavior] += 1
        
        behaviors = [e.behavior for e in events]
        focus_rate = self.calculate_focus_rate(behaviors)
        
        # 找出主要行为
        main_behavior = max(behavior_counts.items(), key=lambda x: x[1])[0] if behavior_counts else 'unknown'
        
        return {
            'person_id': person_id,
            'total_events': len(events),
            'behavior_counts': dict(behavior_counts),
            'focus_rate': round(focus_rate, 3),
            'main_behavior': main_behavior
        }
    
    def get_statistics(self) -> Dict:
        """
        获取完整的统计数据
        
        Returns:
            Dict: 完整统计信息
        """
        # 生成时间段统计
        self.generate_time_segments()
        
        # 计算整体专注度
        all_behaviors = [e.behavior for e in self.events]
        overall_focus_rate = self.calculate_focus_rate(all_behaviors)
        
        # 计算平均专注度趋势
        focus_trend = [seg.focus_rate for seg in self.time_segments]
        avg_focus = sum(focus_trend) / len(focus_trend) if focus_trend else 0
        
        # 找出问题行为最多的时段
        problem_behaviors = ['head_down', 'lying_on_desk']
        problem_segments = []
        for seg in self.time_segments:
            problem_count = sum(seg.behavior_counts.get(b, 0) for b in problem_behaviors)
            if problem_count > 0:
                problem_segments.append({
                    'time': seg.start_time.isoformat(),
                    'problem_count': problem_count
                })
        
        return {
            'summary': {
                'total_events': len(self.events),
                'student_count': self.student_count,
                'duration_seconds': self.get_duration(),
                'overall_focus_rate': round(overall_focus_rate, 3),
                'average_focus_rate': round(avg_focus, 3)
            },
            'behavior_distribution': self.get_behavior_distribution(),
            'time_segments': [
                {
                    'start': seg.start_time.isoformat(),
                    'end': seg.end_time.isoformat(),
                    'person_count': seg.person_count,
                    'focus_rate': round(seg.focus_rate, 3),
                    'behaviors': seg.behavior_counts
                }
                for seg in self.time_segments
            ],
            'problem_periods': sorted(problem_segments, key=lambda x: x['problem_count'], reverse=True)[:5],
            'person_summaries': [
                self.get_person_behavior_summary(pid)
                for pid in range(self.student_count)
            ]
        }
    
    def get_duration(self) -> int:
        """获取检测总时长（秒）"""
        if self.start_time is None or self.end_time is None:
            return 0
        return int((self.end_time - self.start_time).total_seconds())
    
    def get_focus_rate_trend(self) -> List[Tuple[str, float]]:
        """
        获取专注度变化趋势
        
        Returns:
            List[Tuple[str, float]]: [(时间, 专注度), ...]
        """
        return [
            (seg.start_time.strftime('%H:%M:%S'), round(seg.focus_rate, 3))
            for seg in self.time_segments
        ]
    
    def export_json(self, filepath: str):
        """
        导出统计数据为JSON文件
        
        Args:
            filepath: 输出文件路径
        """
        stats = self.get_statistics()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"统计数据已导出: {filepath}")
    
    def reset(self):
        """重置所有统计数据"""
        self.events = []
        self.time_segments = []
        self.total_counts = defaultdict(int)
        self.person_behavior_history = defaultdict(list)
        self.start_time = None
        self.end_time = None
        self.student_count = 0
        logger.info("分析器已重置")
    
    def get_realtime_summary(self) -> Dict:
        """
        获取实时摘要（用于实时显示）
        
        Returns:
            Dict: 实时统计摘要
        """
        if not self.events:
            return {
                'current_persons': 0,
                'current_behaviors': {},
                'focus_rate': 0.0
            }
        
        # 获取最近的事件（最近5秒）
        now = datetime.now()
        recent_events = [
            e for e in self.events
            if (now - e.timestamp).total_seconds() <= 5
        ]
        
        current_behaviors = defaultdict(int)
        for event in recent_events:
            current_behaviors[event.behavior] += 1
        
        behaviors = [e.behavior for e in recent_events]
        current_focus = self.calculate_focus_rate(behaviors)
        
        return {
            'current_persons': len(set(e.person_id for e in recent_events)),
            'current_behaviors': dict(current_behaviors),
            'focus_rate': round(current_focus, 3),
            'total_events': len(self.events)
        }
