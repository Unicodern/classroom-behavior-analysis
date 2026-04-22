"""
数据脱敏模块测试脚本
支持测试本地视频文件和摄像头
可通过参数控制是否启用脱敏
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

from core.privacy_mask import PrivacyMask


def test_video_privacy(video_path: str, 
                       enabled: bool = True,
                       mask_type: str = 'gaussian',
                       display: bool = True):
    """
    测试视频脱敏处理
    
    Args:
        video_path: 视频文件路径
        enabled: 是否启用脱敏
        mask_type: 脱敏类型
        display: 是否实时显示
    """
    logger.info(f"开始测试视频脱敏: {video_path}")
    logger.info(f"脱敏开关: {'启用' if enabled else '禁用'}")
    logger.info(f"脱敏类型: {mask_type}")
    
    # 检查视频文件
    video_file = Path(video_path)
    if not video_file.exists():
        logger.error(f"视频文件不存在: {video_path}")
        return
    
    # 初始化脱敏处理器
    try:
        mask_processor = PrivacyMask(
            enabled=enabled,
            mask_type=mask_type,
            kernel_size=51,
            pixel_size=10
        )
        logger.info("脱敏处理器初始化成功")
    except Exception as e:
        logger.error(f"脱敏处理器初始化失败: {e}")
        return
    
    # 设置输出路径
    output_dir = project_root / 'output' / 'videos'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_masked_{mask_type}" if enabled else "_original"
    output_path = str(output_dir / f"{video_file.stem}{suffix}.mp4")
    
    # 处理视频
    try:
        stats = mask_processor.process_video(
            source=str(video_file),
            output_path=output_path,
            display=display
        )
        
        logger.info("=" * 50)
        logger.info("处理完成!")
        logger.info(f"总帧数: {stats['processed_frames']}")
        logger.info(f"脱敏面部数: {stats['masked_faces']}")
        logger.info(f"输出视频: {output_path}")
        
    except KeyboardInterrupt:
        logger.info("处理被用户中断")
    except Exception as e:
        logger.error(f"处理失败: {e}")


def test_camera_privacy(camera_id: int = 0,
                        enabled: bool = True,
                        mask_type: str = 'gaussian',
                        duration: int = 30):
    """
    测试摄像头实时脱敏
    
    Args:
        camera_id: 摄像头索引
        enabled: 是否启用脱敏
        mask_type: 脱敏类型
        duration: 测试时长（秒）
    """
    logger.info(f"开始测试摄像头脱敏 (ID: {camera_id})")
    logger.info(f"脱敏开关: {'启用' if enabled else '禁用'}")
    logger.info(f"按 'q' 键退出")
    
    # 初始化脱敏处理器
    try:
        mask_processor = PrivacyMask(
            enabled=enabled,
            mask_type=mask_type
        )
        logger.info("脱敏处理器初始化成功")
    except Exception as e:
        logger.error(f"脱敏处理器初始化失败: {e}")
        return
    
    # 设置输出路径
    output_dir = project_root / 'output' / 'videos'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"camera_masked_{mask_type}" if enabled else "camera_original"
    output_path = str(output_dir / f"{suffix}.mp4")
    
    import time
    start_time = time.time()
    
    try:
        stats = mask_processor.process_video(
            source=camera_id,
            output_path=output_path,
            display=True
        )
        
        # 检查时长
        if time.time() - start_time >= duration:
            logger.info(f"已达到测试时长 {duration} 秒")
            
    except KeyboardInterrupt:
        logger.info("处理被用户中断")
    
    logger.info("=" * 50)
    logger.info("摄像头测试完成!")


def compare_mask_types(video_path: str):
    """
    对比不同脱敏类型的效果
    
    Args:
        video_path: 视频文件路径
    """
    logger.info("对比不同脱敏类型效果")
    
    video_file = Path(video_path)
    if not video_file.exists():
        logger.error(f"视频文件不存在: {video_path}")
        return
    
    mask_types = ['gaussian', 'pixelate', 'mosaic', 'solid']
    
    for mask_type in mask_types:
        logger.info(f"\n测试脱敏类型: {mask_type}")
        
        mask_processor = PrivacyMask(
            enabled=True,
            mask_type=mask_type
        )
        
        output_dir = project_root / 'output' / 'videos'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"compare_{mask_type}.mp4")
        
        try:
            stats = mask_processor.process_video(
                source=str(video_file),
                output_path=output_path,
                display=False  # 不显示，快速处理
            )
            
            logger.info(f"  完成: {stats['processed_frames']}帧, "
                       f"脱敏{stats['masked_faces']}个面部")
            logger.info(f"  输出: {output_path}")
            
        except Exception as e:
            logger.error(f"  失败: {e}")
    
    logger.info("\n对比完成! 请查看 output/videos/ 目录下的视频文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='数据脱敏模块测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 测试视频脱敏（启用，高斯模糊）
  python test_privacy.py --video test_video.mp4
  
  # 测试视频但不脱敏（对比效果）
  python test_privacy.py --video test_video.mp4 --no-mask
  
  # 使用像素化脱敏
  python test_privacy.py --video test_video.mp4 --type pixelate
  
  # 使用马赛克脱敏
  python test_privacy.py --video test_video.mp4 --type mosaic
  
  # 测试摄像头实时脱敏
  python test_privacy.py --camera
  
  # 对比所有脱敏类型
  python test_privacy.py --video test_video.mp4 --compare
        """
    )
    
    parser.add_argument('--video', '-v', type=str,
                       help='测试本地视频文件路径')
    parser.add_argument('--camera', '-c', action='store_true',
                       help='测试摄像头')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='摄像头索引（默认0）')
    parser.add_argument('--no-mask', action='store_true',
                       help='禁用脱敏（用于对比）')
    parser.add_argument('--type', '-t', type=str, default='gaussian',
                       choices=['gaussian', 'pixelate', 'mosaic', 'solid'],
                       help='脱敏类型（默认gaussian）')
    parser.add_argument('--duration', '-d', type=int, default=30,
                       help='摄像头测试时长（秒，默认30）')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示实时画面（后台模式）')
    parser.add_argument('--compare', action='store_true',
                       help='对比所有脱敏类型')
    
    args = parser.parse_args()
    
    # 对比模式
    if args.compare and args.video:
        compare_mask_types(args.video)
        return
    
    # 确定是否启用脱敏
    enabled = not args.no_mask
    
    # 测试视频文件
    if args.video:
        test_video_privacy(
            video_path=args.video,
            enabled=enabled,
            mask_type=args.type,
            display=not args.no_display
        )
    
    # 测试摄像头
    elif args.camera:
        test_camera_privacy(
            camera_id=args.camera_id,
            enabled=enabled,
            mask_type=args.type,
            duration=args.duration
        )
    
    # 默认测试本地视频
    else:
        default_video = project_root / 'test_video.mp4'
        if default_video.exists():
            logger.info(f"使用默认视频文件: {default_video}")
            test_video_privacy(
                video_path=str(default_video),
                enabled=enabled,
                mask_type=args.type,
                display=not args.no_display
            )
        else:
            logger.error(f"默认视频文件不存在: {default_video}")
            logger.info("请使用 --video 指定视频文件，或使用 --camera 测试摄像头")
            parser.print_help()


if __name__ == '__main__':
    main()
