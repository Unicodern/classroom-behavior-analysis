"""
行为检测模块测试脚本
支持测试本地视频文件和摄像头
"""

import sys
import argparse
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.detector import BehaviorDetector
from core.behavior_analyzer import BehaviorAnalyzer


def test_video_file(video_path: str, save_output: bool = True, display: bool = True):
    """
    测试本地视频文件
    
    Args:
        video_path: 视频文件路径
        save_output: 是否保存处理后的视频
        display: 是否实时显示
    """
    logger.info(f"开始测试视频文件: {video_path}")
    
    # 检查视频文件是否存在
    video_file = Path(video_path)
    if not video_file.exists():
        logger.error(f"视频文件不存在: {video_path}")
        return
    
    # 初始化检测器
    try:
        detector = BehaviorDetector(conf_threshold=0.5)
        logger.info("检测器初始化成功")
    except Exception as e:
        logger.error(f"检测器初始化失败: {e}")
        return
    
    # 初始化分析器
    analyzer = BehaviorAnalyzer(segment_duration=10)  # 10秒一个时间段
    
    # 设置输出路径
    save_path = None
    if save_output:
        output_dir = project_root / 'output' / 'videos'
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(output_dir / f"detected_{video_file.name}")
    
    # 处理视频
    frame_count = 0
    try:
        for result in detector.process_video(
            source=str(video_file),
            display=display,
            save_path=save_path
        ):
            frame_count += 1
            
            # 更新分析器
            analyzer.update(
                behaviors=result['behaviors'],
                timestamp=result.get('timestamp')
            )
            
            # 每30帧打印一次统计
            if frame_count % 30 == 0:
                summary = analyzer.get_realtime_summary()
                logger.info(
                    f"帧 {result['frame_id']}: "
                    f"检测到 {summary['current_persons']} 人, "
                    f"专注度: {summary['focus_rate']:.2%}"
                )
    
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    
    # 输出最终统计
    logger.info("=" * 50)
    logger.info("检测完成，最终统计:")
    logger.info(f"总处理帧数: {frame_count}")
    
    stats = analyzer.get_statistics()
    logger.info(f"总事件数: {stats['summary']['total_events']}")
    logger.info(f"检测学生数: {stats['summary']['student_count']}")
    logger.info(f"整体专注度: {stats['summary']['overall_focus_rate']:.2%}")
    
    # 行为分布
    logger.info("\n行为分布:")
    for behavior, data in stats['behavior_distribution'].items():
        logger.info(f"  {behavior}: {data['count']}次 ({data['percentage']}%)")
    
    # 保存统计结果
    import json
    from datetime import datetime
    
    # 创建报告目录
    report_dir = project_root / 'output' / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名（基于视频名+时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = video_file.stem
    
    # 1. 保存完整统计报告
    stats_path = report_dir / f"{video_name}_stats_{timestamp}.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"\n完整统计报告: {stats_path}")
    
    # 2. 保存Kimi分析专用格式（简化版）
    kimi_data = {
        'video_name': video_file.name,
        'analysis_time': timestamp,
        'duration_seconds': stats['summary'].get('duration_seconds', 0),
        'total_frames': frame_count,
        'total_persons': stats['summary']['student_count'],
        'behaviors': {
            name: data['count'] 
            for name, data in stats['behavior_distribution'].items()
        },
        'focus_rate': stats['summary']['overall_focus_rate']
    }
    
    kimi_path = report_dir / f"{video_name}_kimi_{timestamp}.json"
    with open(kimi_path, 'w', encoding='utf-8') as f:
        json.dump(kimi_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Kimi分析数据: {kimi_path}")
    
    if save_path:
        logger.info(f"处理后的视频: {save_path}")
    
    return stats, kimi_data


def test_camera(camera_id: int = 0, duration: int = 30):
    """
    测试摄像头
    
    Args:
        camera_id: 摄像头索引，默认0
        duration: 测试时长（秒），默认30秒
    """
    logger.info(f"开始测试摄像头 (ID: {camera_id})")
    logger.info(f"按 'q' 键退出，将测试 {duration} 秒或直到手动退出")
    
    # 初始化检测器
    try:
        detector = BehaviorDetector(conf_threshold=0.5)
        logger.info("检测器初始化成功")
    except Exception as e:
        logger.error(f"检测器初始化失败: {e}")
        return
    
    # 初始化分析器
    analyzer = BehaviorAnalyzer(segment_duration=5)
    
    # 处理摄像头流
    import time
    start_time = time.time()
    frame_count = 0
    
    try:
        for result in detector.process_video(
            source=camera_id,
            display=True,
            save_path=None
        ):
            frame_count += 1
            
            # 更新分析器
            analyzer.update(
                behaviors=result['behaviors'],
                timestamp=result.get('timestamp')
            )
            
            # 每10帧打印一次
            if frame_count % 10 == 0:
                summary = analyzer.get_realtime_summary()
                logger.info(
                    f"帧 {result['frame_id']}: "
                    f"{summary['current_persons']}人, "
                    f"专注度: {summary['focus_rate']:.2%}"
                )
            
            # 检查是否达到测试时长
            if time.time() - start_time >= duration:
                logger.info(f"已达到测试时长 {duration} 秒")
                break
    
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    
    # 输出统计
    logger.info("=" * 50)
    logger.info("摄像头测试完成")
    logger.info(f"总处理帧数: {frame_count}")
    
    stats = analyzer.get_statistics()
    logger.info(f"整体专注度: {stats['summary']['overall_focus_rate']:.2%}")
    
    logger.info("\n行为分布:")
    for behavior, data in stats['behavior_distribution'].items():
        logger.info(f"  {behavior}: {data['count']}次")


def list_available_cameras(max_check: int = 5):
    """
    列出可用的摄像头设备
    
    Args:
        max_check: 最大检查的摄像头数量
    """
    import cv2
    
    logger.info("检查可用摄像头...")
    available = []
    
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                available.append({
                    'id': i,
                    'resolution': f"{width}x{height}",
                    'fps': fps
                })
                logger.info(f"  摄像头 {i}: {width}x{height} @ {fps}fps")
            cap.release()
    
    if not available:
        logger.warning("未找到可用摄像头")
    else:
        logger.info(f"共找到 {len(available)} 个可用摄像头")
    
    return available


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='课堂行为检测模块测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 测试本地视频文件
  python test_detector.py --video test_video.mp4
  
  # 测试摄像头（默认摄像头，30秒）
  python test_detector.py --camera
  
  # 测试指定摄像头60秒
  python test_detector.py --camera --camera-id 1 --duration 60
  
  # 只处理不显示（后台模式）
  python test_detector.py --video test_video.mp4 --no-display
  
  # 列出可用摄像头
  python test_detector.py --list-cameras
        """
    )
    
    parser.add_argument('--video', '-v', type=str,
                       help='测试本地视频文件路径')
    parser.add_argument('--camera', '-c', action='store_true',
                       help='测试摄像头')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='摄像头索引（默认0）')
    parser.add_argument('--duration', '-d', type=int, default=30,
                       help='摄像头测试时长（秒，默认30）')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示实时画面（后台模式）')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存输出视频')
    parser.add_argument('--list-cameras', '-l', action='store_true',
                       help='列出可用摄像头')
    
    args = parser.parse_args()
    
    # 列出摄像头
    if args.list_cameras:
        list_available_cameras()
        return
    
    # 测试视频文件
    if args.video:
        test_video_file(
            video_path=args.video,
            save_output=not args.no_save,
            display=not args.no_display
        )
    
    # 测试摄像头
    elif args.camera:
        test_camera(
            camera_id=args.camera_id,
            duration=args.duration
        )
    
    # 默认测试本地视频
    else:
        default_video = project_root / 'test_video.mp4'
        if default_video.exists():
            logger.info(f"使用默认视频文件: {default_video}")
            test_video_file(
                video_path=str(default_video),
                save_output=not args.no_save,
                display=not args.no_display
            )
        else:
            logger.error(f"默认视频文件不存在: {default_video}")
            logger.info("请使用 --video 指定视频文件，或使用 --camera 测试摄像头")
            parser.print_help()


if __name__ == '__main__':
    main()
