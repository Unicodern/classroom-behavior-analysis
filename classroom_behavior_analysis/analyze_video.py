"""
端到端视频分析流程
实现：视频检测 → 数据统计 → Kimi分析 → 报告生成
一键完成课堂行为智能分析

# 完整流程（检测 + 统计 + Kimi分析）
python analyze_video.py --video test_video.mp4

# 不使用Kimi分析（仅检测统计）
python analyze_video.py --video test_video.mp4 --no-kimi

# 实时显示检测过程
python analyze_video.py --video test_video.mp4 --display

# 📹 分析本地视频文件
python analyze_video.py --video test_video.mp4
python analyze_video.py --video test_video.mp4 --display

# 📷 使用摄像头实时检测（默认30秒）
python analyze_video.py --camera 0

# 📷 使用摄像头检测60秒
python analyze_video.py --camera 0 --duration 60

# 📷 使用摄像头并实时显示
python analyze_video.py --camera 0 --display

# 📷 使用摄像头并保存视频
python analyze_video.py --camera 0 --display --duration 10 --display

"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载环境变量（从.env文件）
from dotenv import load_dotenv
load_dotenv()

from core.detector import BehaviorDetector
from core.behavior_analyzer import BehaviorAnalyzer
from api.kimi_client import KimiClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """
    视频分析器（端到端）
    
    完整流程：
    1. 视频行为检测（YOLOv8-pose）
    2. 数据统计与保存
    3. Kimi AI智能分析
    4. 生成可视化报告
    """
    
    def __init__(self, conf_threshold: float = 0.5):
        """
        初始化分析器
        
        Args:
            conf_threshold: 检测置信度阈值
        """
        self.conf_threshold = conf_threshold
        
        # 初始化检测器
        self.detector = BehaviorDetector(
            conf_threshold=conf_threshold
        )
        
        # 初始化分析器
        self.analyzer = BehaviorAnalyzer(segment_duration=10)
        
        # 初始化Kimi客户端（可选）
        self.kimi_client = None
        api_key = os.getenv('KIMI_API_KEY') or os.getenv('MOONSHOT_API_KEY')
        
        # 调试信息
        if api_key:
            logger.info(f"检测到API密钥: {api_key[:10]}...")
            try:
                self.kimi_client = KimiClient(api_key=api_key)
                logger.info("Kimi客户端初始化成功")
            except Exception as e:
                logger.warning(f"Kimi客户端初始化失败: {e}")
        else:
            logger.warning("未检测到API密钥（KIMI_API_KEY 或 MOONSHOT_API_KEY），跳过Kimi分析")
        
        # 创建基础输出目录
        self.output_dir = project_root / 'output'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("视频分析器初始化完成")
    
    def _create_session_dirs(self, video_name: str) -> Dict[str, Path]:
        """
        创建本次分析的目录结构
        
        结构: output/YYYY-MM-DD/video_name_HHMMSS/
            ├── video/      - 处理后的视频
            ├── reports/    - 统计报告和Kimi分析
            └── final_report.json
        
        Args:
            video_name: 视频名称
            
        Returns:
            Dict[str, Path]: 各目录路径
        """
        from datetime import datetime
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%H%M%S")
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # 创建目录结构: output/日期/video_name_时间戳/
        session_dir = self.output_dir / date_str / f"{video_name}_{timestamp}"
        video_dir = session_dir / 'video'
        report_dir = session_dir / 'reports'
        
        # 创建所有目录
        video_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"创建分析目录: {session_dir}")
        
        return {
            'session': session_dir,
            'video': video_dir,
            'reports': report_dir
        }
    
    def analyze(self, video_path: str = None, camera_id: int = None,
                display: bool = False, save_video: bool = True, 
                use_kimi: bool = True, duration: int = 30) -> Dict:
        """
        分析视频或摄像头（完整流程）
        
        Args:
            video_path: 视频文件路径（与camera_id二选一）
            camera_id: 摄像头ID（与video_path二选一）
            display: 是否实时显示
            save_video: 是否保存处理后的视频
            use_kimi: 是否使用Kimi进行AI分析
            duration: 摄像头测试时长（秒），默认30秒
            
        Returns:
            Dict: 完整分析结果
        """
        # 确定视频源
        if camera_id is not None:
            source = camera_id
            video_name = f"camera_{camera_id}"
            # 为摄像头模式创建一个虚拟的Path对象用于保存文件
            video_file = self.output_dir / f"{video_name}.mp4"
            logger.info(f"使用摄像头: {camera_id}")
        elif video_path:
            video_file = Path(video_path)
            if not video_file.exists():
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            source = str(video_file)
            video_name = video_file.stem
        else:
            raise ValueError("请提供video_path或camera_id")
        
        # 创建本次分析的目录结构
        self.session_dirs = self._create_session_dirs(video_name)
        
        logger.info(f"开始分析: {video_name}")
        logger.info(f"Kimi分析: {'开启' if use_kimi and self.kimi_client else '关闭'}")
        
        # 步骤1: 行为检测
        logger.info("\n" + "="*50)
        logger.info("步骤1/3: 行为检测")
        logger.info("="*50)
        
        if camera_id is not None:
            stats_data = self._detect_camera(camera_id, display, save_video, duration)
        else:
            stats_data = self._detect_video(video_file, display, save_video)
        
        # 步骤2: 保存统计数据
        logger.info("\n" + "="*50)
        logger.info("步骤2/3: 保存统计数据")
        logger.info("="*50)
        
        saved_files = self._save_statistics(video_file, stats_data)
        
        # 步骤3: Kimi分析（可选）
        kimi_result = None
        if use_kimi and self.kimi_client:
            logger.info("\n" + "="*50)
            logger.info("步骤3/3: Kimi智能分析")
            logger.info("="*50)
            
            kimi_result = self._analyze_with_kimi(stats_data)
            
            # 保存Kimi分析结果
            if kimi_result:
                kimi_path = saved_files['kimi_analysis']
                with open(kimi_path, 'w', encoding='utf-8') as f:
                    json.dump(kimi_result, f, ensure_ascii=False, indent=2)
                logger.info(f"Kimi分析报告: {kimi_path}")
        
        # 生成最终报告
        final_report = self._generate_final_report(
            video_file, stats_data, kimi_result, saved_files
        )
        
        logger.info("\n" + "="*50)
        logger.info("分析完成!")
        logger.info("="*50)
        logger.info(f"视频: {video_file.name}")
        logger.info(f"检测帧数: {stats_data['total_frames']}")
        logger.info(f"学生人数: {stats_data['total_persons']}")
        logger.info(f"整体专注度: {stats_data['focus_rate']:.2%}")
        if kimi_result:
            logger.info(f"课堂评价: {kimi_result.get('summary', 'N/A')[:50]}...")
        
        return final_report
    
    def _detect_video(self, video_file: Path, display: bool, 
                      save_video: bool) -> Dict:
        """
        检测视频并收集统计数据
        
        Args:
            video_file: 视频文件路径
            display: 是否显示
            save_video: 是否保存视频
            
        Returns:
            Dict: 统计数据
        """
        # 设置保存路径 - 使用session的视频目录
        save_path = None
        if save_video:
            save_path = str(self.session_dirs['video'] / f"detected_{video_file.name}")
        
        # 重置分析器
        self.analyzer = BehaviorAnalyzer(segment_duration=10)
        
        # 处理视频
        frame_count = 0
        all_behaviors = []
        
        for result in self.detector.process_video(
            source=str(video_file),
            display=display,
            save_path=save_path
        ):
            frame_count += 1
            
            # 更新分析器
            self.analyzer.update(
                behaviors=result['behaviors'],
                timestamp=result.get('timestamp')
            )
            
            # 收集所有行为数据
            all_behaviors.extend(result['behaviors'])
            
            # 每100帧打印进度
            if frame_count % 100 == 0:
                summary = self.analyzer.get_realtime_summary()
                logger.info(f"  已处理 {frame_count} 帧, "
                           f"检测 {summary['current_persons']} 人")
        
        # 获取统计结果
        stats = self.analyzer.get_statistics()
        
        # 构建返回数据
        behavior_counts = {
            name: data['count']
            for name, data in stats['behavior_distribution'].items()
        }
        
        return {
            'total_frames': frame_count,
            'total_persons': stats['summary']['student_count'],
            'behaviors': behavior_counts,
            'focus_rate': stats['summary']['overall_focus_rate'],
            'duration_seconds': stats['summary'].get('duration_seconds', 0),
            'raw_statistics': stats,
            'all_behaviors': all_behaviors
        }
    
    def _detect_camera(self, camera_id: int, display: bool, 
                       save_video: bool, duration: int) -> Dict:
        """
        检测摄像头并收集统计数据
        
        Args:
            camera_id: 摄像头ID
            display: 是否显示
            save_video: 是否保存视频
            duration: 检测时长（秒）
            
        Returns:
            Dict: 统计数据
        """
        import time
        
        # 设置保存路径 - 使用session的视频目录
        save_path = None
        if save_video:
            save_path = str(self.session_dirs['video'] / f"camera_{camera_id}_live.mp4")
        
        # 重置分析器
        self.analyzer = BehaviorAnalyzer(segment_duration=10)
        
        # 处理摄像头
        frame_count = 0
        all_behaviors = []
        start_time = time.time()
        
        # 显示提示信息
        if duration > 0:
            logger.info(f"摄像头检测将持续 {duration} 秒，按 'q' 键可提前结束")
        else:
            logger.info("摄像头检测将持续运行，按 'q' 键结束")
        
        # 使用生成器并确保正确关闭
        video_generator = self.detector.process_video(
            source=camera_id,
            display=display,
            save_path=save_path
        )
        
        try:
            for result in video_generator:
                frame_count += 1
                
                # 更新分析器
                self.analyzer.update(
                    behaviors=result['behaviors'],
                    timestamp=result.get('timestamp')
                )
                
                # 收集所有行为数据
                all_behaviors.extend(result['behaviors'])
                
                # 每30帧打印进度
                if frame_count % 30 == 0:
                    summary = self.analyzer.get_realtime_summary()
                    elapsed = time.time() - start_time
                    time_info = f"{elapsed:.1f}秒" if duration == 0 else f"{elapsed:.1f}/{duration}秒"
                    logger.info(f"  已处理 {frame_count} 帧, "
                               f"检测 {summary['current_persons']} 人, "
                               f"已运行 {time_info}")
                
                # 检查是否达到时长（duration=0表示无限时长）
                if duration > 0 and time.time() - start_time >= duration:
                    logger.info(f"已达到设定时长 {duration} 秒")
                    break
        finally:
            # 确保生成器正确关闭，释放摄像头资源
            video_generator.close()
            logger.info("摄像头资源已释放")
        
        # 获取统计结果
        stats = self.analyzer.get_statistics()
        actual_duration = time.time() - start_time
        
        # 构建返回数据
        behavior_counts = {
            name: data['count']
            for name, data in stats['behavior_distribution'].items()
        }
        
        return {
            'total_frames': frame_count,
            'total_persons': stats['summary']['student_count'],
            'behaviors': behavior_counts,
            'focus_rate': stats['summary']['overall_focus_rate'],
            'duration_seconds': actual_duration,
            'raw_statistics': stats,
            'all_behaviors': all_behaviors
        }
    
    def _save_statistics(self, video_file: Path, 
                         stats_data: Dict) -> Dict[str, Path]:
        """
        保存统计数据到文件
        
        Args:
            video_file: 视频文件
            stats_data: 统计数据
            
        Returns:
            Dict[str, Path]: 保存的文件路径
        """
        video_name = video_file.stem
        report_dir = self.session_dirs['reports']
        
        saved_files = {}
        
        # 1. 完整统计报告
        full_stats = {
            'video_info': {
                'name': video_file.name,
                'analysis_time': datetime.now().isoformat(),
            },
            'summary': {
                'total_frames': stats_data['total_frames'],
                'total_persons': stats_data['total_persons'],
                'duration_seconds': stats_data['duration_seconds'],
                'focus_rate': stats_data['focus_rate']
            },
            'behaviors': stats_data['behaviors'],
            'detailed_statistics': stats_data['raw_statistics']
        }
        
        stats_path = report_dir / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(full_stats, f, ensure_ascii=False, indent=2)
        saved_files['full_stats'] = stats_path
        logger.info(f"完整统计: {stats_path}")
        
        # 2. Kimi专用格式
        kimi_data = {
            'video_name': video_file.name,
            'analysis_time': datetime.now().isoformat(),
            'duration_seconds': stats_data['duration_seconds'],
            'total_frames': stats_data['total_frames'],
            'total_persons': stats_data['total_persons'],
            'behaviors': stats_data['behaviors'],
            'focus_rate': stats_data['focus_rate']
        }
        
        kimi_path = report_dir / "kimi_data.json"
        with open(kimi_path, 'w', encoding='utf-8') as f:
            json.dump(kimi_data, f, ensure_ascii=False, indent=2)
        saved_files['kimi_data'] = kimi_path
        logger.info(f"Kimi数据: {kimi_path}")
        
        # 3. 保存Kimi分析结果的路径（预留）
        saved_files['kimi_analysis'] = report_dir / "kimi_analysis.json"
        
        return saved_files
    
    def _analyze_with_kimi(self, stats_data: Dict) -> Optional[Dict]:
        """
        使用Kimi分析统计数据
        
        Args:
            stats_data: 统计数据
            
        Returns:
            Optional[Dict]: Kimi分析结果
        """
        if not self.kimi_client:
            logger.warning("Kimi客户端未初始化，跳过AI分析")
            return None
        
        try:
            # 构建分析数据
            behavior_data = {
                'total_frames': stats_data['total_frames'],
                'total_persons': stats_data['total_persons'],
                'behaviors': stats_data['behaviors'],
                'focus_rate': stats_data['focus_rate'],
                'duration': stats_data['duration_seconds']
            }
            
            logger.info("正在调用Kimi API进行智能分析...")
            result = self.kimi_client.analyze_behavior(behavior_data)
            
            logger.info("Kimi分析完成")
            logger.info(f"  课堂评价: {result.get('summary', 'N/A')[:60]}...")
            logger.info(f"  专注度等级: {result.get('focus_level', 'N/A')}")
            logger.info(f"  参与度评分: {result.get('engagement_score', 0)}")
            
            if result.get('suggestions'):
                logger.info(f"  改进建议: {len(result['suggestions'])} 条")
            
            return result
            
        except Exception as e:
            logger.error(f"Kimi分析失败: {e}")
            return None
    
    def _generate_final_report(self, video_file: Path, stats_data: Dict,
                               kimi_result: Optional[Dict],
                               saved_files: Dict) -> Dict:
        """
        生成最终综合报告
        
        Args:
            video_file: 视频文件
            stats_data: 统计数据
            kimi_result: Kimi分析结果
            saved_files: 保存的文件路径
            
        Returns:
            Dict: 综合报告
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'analysis_info': {
                'video_name': video_file.name,
                'analysis_time': datetime.now().isoformat()
            },
            'statistics': {
                'total_frames': stats_data['total_frames'],
                'total_persons': stats_data['total_persons'],
                'duration_seconds': stats_data['duration_seconds'],
                'focus_rate': stats_data['focus_rate'],
                'behaviors': stats_data['behaviors']
            },
            'kimi_analysis': kimi_result,
            'output_files': {
                key: str(path) for key, path in saved_files.items()
            }
        }
        
        # 保存综合报告到session目录
        report_path = self.session_dirs['session'] / "final_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"综合报告: {report_path}")
        logger.info(f"所有文件保存在: {self.session_dirs['session']}")
        
        return report


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='课堂行为智能分析系统 - 端到端分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 分析视频文件（完整流程）
  python analyze_video.py --video test_video.mp4
  
  # 使用摄像头实时检测（默认45分钟=一节课时长）
  python analyze_video.py --camera 0
  
  # 使用摄像头检测10分钟
  python analyze_video.py --camera 0 --duration 600
  
  # 使用摄像头持续运行（按q键结束）
  python analyze_video.py --camera 0 --duration 0
  
  # 不使用Kimi分析（仅检测统计）
  python analyze_video.py --video test_video.mp4 --no-kimi
  
  # 实时显示检测过程
  python analyze_video.py --video test_video.mp4 --display
        """
    )
    
    parser.add_argument('--video', '-v', type=str,
                       help='要分析的视频文件路径')
    parser.add_argument('--camera', '-c', type=int, metavar='ID',
                       help='使用摄像头进行检测（指定摄像头ID，默认0）')
    parser.add_argument('--duration', '-d', type=int, default=2700,
                       help='摄像头检测时长（秒，默认2700=45分钟，0表示无限时长直到按q键）')
    parser.add_argument('--display', action='store_true',
                       help='实时显示检测过程')
    parser.add_argument('--no-save-video', action='store_true',
                       help='不保存处理后的视频')
    parser.add_argument('--no-kimi', action='store_true',
                       help='不使用Kimi进行AI分析')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='检测置信度阈值（默认0.5）')
    
    args = parser.parse_args()
    
    # 检查输入参数
    if not args.video and args.camera is None:
        logger.error("请提供--video或--camera参数")
        parser.print_help()
        return
    
    # 创建分析器
    analyzer = VideoAnalyzer(
        conf_threshold=args.conf
    )
    
    # 执行分析
    try:
        if args.camera is not None:
            # 摄像头模式
            result = analyzer.analyze(
                camera_id=args.camera,
                display=args.display,
                save_video=not args.no_save_video,
                use_kimi=not args.no_kimi,
                duration=args.duration
            )
        else:
            # 视频文件模式
            video_path = Path(args.video)
            if not video_path.exists():
                # 尝试在项目根目录查找
                video_path = project_root / args.video
                if not video_path.exists():
                    logger.error(f"视频文件不存在: {args.video}")
                    return
            
            result = analyzer.analyze(
                video_path=str(video_path),
                display=args.display,
                save_video=not args.no_save_video,
                use_kimi=not args.no_kimi
            )
        
        logger.info("\n分析流程全部完成!")
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise


if __name__ == '__main__':
    main()
