"""
课堂行为智能分析系统 - Web应用
Flask后端：提供视频检测、摄像头实时检测、报告查看、API Key设置等功能
"""

import os
import sys
import json
import time
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
load_dotenv()

from core.detector import BehaviorDetector
from core.behavior_analyzer import BehaviorAnalyzer
from api.kimi_client import KimiClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(project_root / 'uploads')

UPLOAD_FOLDER = Path(app.config['UPLOAD_FOLDER'])
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = project_root / 'output'

detector = BehaviorDetector(conf_threshold=0.5)

api_key_store = {'key': None}

camera_state = {
    'running': False,
    'lock': threading.Lock(),
    'frame_count': 0,
    'analyzer': None,
    'session_dirs': None,
    'start_time': None,
    'stop_event': threading.Event(),
    'latest_frame': None,
    'latest_frame_lock': threading.Lock()
}

video_task_state = {
    'lock': threading.Lock(),
    'progress': None,
    'latest_frame': None,
    'latest_frame_lock': threading.Lock(),
    'result': None,
    'session_dirs': None
}


def get_kimi_client() -> Optional[KimiClient]:
    key = api_key_store['key'] or os.getenv('KIMI_API_KEY') or os.getenv('MOONSHOT_API_KEY')
    if not key:
        return None
    try:
        return KimiClient(api_key=key)
    except Exception as e:
        logger.error(f"Kimi客户端初始化失败: {e}")
        return None


def create_session_dirs(video_name: str) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%H%M%S")
    date_str = datetime.now().strftime("%Y-%m-%d")
    session_dir = OUTPUT_DIR / date_str / f"{video_name}_{timestamp}"
    video_dir = session_dir / 'video'
    report_dir = session_dir / 'reports'
    video_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return {'session': session_dir, 'video': video_dir, 'reports': report_dir}


def do_kimi_analysis(stats_data: Dict, session_dirs: Dict) -> Optional[Dict]:
    kimi_client = get_kimi_client()
    if not kimi_client:
        logger.warning("Kimi客户端未初始化，跳过AI分析")
        return None
    try:
        behavior_data = {
            'total_frames': stats_data['total_frames'],
            'total_persons': stats_data['total_persons'],
            'behaviors': stats_data['behaviors'],
            'focus_rate': stats_data['focus_rate'],
            'duration': stats_data['duration_seconds']
        }
        logger.info("正在调用Kimi API进行智能分析...")
        result = kimi_client.analyze_behavior(behavior_data)
        if result:
            kimi_path = session_dirs['reports'] / "kimi_analysis.json"
            with open(kimi_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Kimi分析报告已保存: {kimi_path}")
        return result
    except Exception as e:
        logger.error(f"Kimi分析失败: {e}")
        return None


def save_statistics(stats_data: Dict, session_dirs: Dict, video_name: str):
    report_dir = session_dirs['reports']
    full_stats = {
        'video_info': {
            'name': video_name,
            'analysis_time': datetime.now().isoformat(),
        },
        'summary': {
            'total_frames': stats_data['total_frames'],
            'total_persons': stats_data['total_persons'],
            'duration_seconds': stats_data['duration_seconds'],
            'focus_rate': stats_data['focus_rate']
        },
        'behaviors': stats_data['behaviors'],
        'detailed_statistics': stats_data.get('raw_statistics', {})
    }
    stats_path = report_dir / "statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(full_stats, f, ensure_ascii=False, indent=2)

    kimi_data = {
        'video_name': video_name,
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


def generate_final_report(stats_data: Dict, kimi_result: Optional[Dict],
                          session_dirs: Dict, video_name: str) -> Dict:
    report = {
        'analysis_info': {
            'video_name': video_name,
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
            'session_dir': str(session_dirs['session']),
            'video_dir': str(session_dirs['video']),
            'reports_dir': str(session_dirs['reports'])
        }
    }
    report_path = session_dirs['session'] / "final_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"综合报告已保存: {report_path}")
    return report


# ========== 视频检测后台任务 ==========

def video_detection_task(video_path: str, session_dirs: Dict, video_name: str):
    with video_task_state['lock']:
        video_task_state['progress'] = 'detecting'
        video_task_state['result'] = None

    analyzer = BehaviorAnalyzer(segment_duration=10)
    save_path = str(session_dirs['video'] / f"detected_{video_name}.mp4")
    frame_count = 0
    all_behaviors = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with video_task_state['lock']:
            video_task_state['progress'] = 'error'
            video_task_state['result'] = {'error': '无法打开视频文件'}
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            persons = detector.detect(frame)
            behaviors = []
            vis_frame = frame.copy()

            for person in persons:
                behavior, conf = detector.classify_behavior(person)
                behaviors.append({
                    'person_id': person.person_id,
                    'behavior': behavior,
                    'confidence': conf
                })
                vis_frame = detector._draw_detection(vis_frame, person, behavior, conf)

            info_text = f"Frame: {frame_count}/{total_frames} | Persons: {len(persons)}"
            cv2.putText(vis_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            writer.write(vis_frame)

            analyzer.update(behaviors=behaviors, timestamp=datetime.now())
            all_behaviors.extend(behaviors)

            _, jpeg = cv2.imencode('.jpg', vis_frame)
            with video_task_state['latest_frame_lock']:
                video_task_state['latest_frame'] = jpeg.tobytes()

            if frame_count % 50 == 0:
                logger.info(f"视频检测进度: {frame_count}/{total_frames} 帧")

        writer.release()
        cap.release()

        stats = analyzer.get_statistics()
        behavior_counts = {
            name: data['count']
            for name, data in stats['behavior_distribution'].items()
        }
        duration_seconds = stats['summary'].get('duration_seconds', 0)
        if duration_seconds == 0 and fps > 0:
            duration_seconds = frame_count / fps

        stats_data = {
            'total_frames': frame_count,
            'total_persons': stats['summary']['student_count'],
            'behaviors': behavior_counts,
            'focus_rate': stats['summary']['overall_focus_rate'],
            'duration_seconds': duration_seconds,
            'raw_statistics': stats,
            'all_behaviors': all_behaviors
        }

        save_statistics(stats_data, session_dirs, video_name)

        with video_task_state['lock']:
            video_task_state['progress'] = 'analyzing'

        kimi_result = do_kimi_analysis(stats_data, session_dirs)
        final_report = generate_final_report(stats_data, kimi_result, session_dirs, video_name)

        with video_task_state['lock']:
            video_task_state['progress'] = 'done'
            video_task_state['result'] = final_report

        logger.info(f"视频检测完成: {frame_count} 帧")

    except Exception as e:
        logger.error(f"视频检测出错: {e}")
        with video_task_state['lock']:
            video_task_state['progress'] = 'error'
            video_task_state['result'] = {'error': str(e)}
    finally:
        if cap.isOpened():
            cap.release()
        if writer:
            writer.release()


# ========== 摄像头检测后台任务 ==========

def camera_detection_task(camera_id: int, session_dirs: Dict):
    import cv2

    with camera_state['lock']:
        camera_state['running'] = True
        camera_state['frame_count'] = 0
        camera_state['start_time'] = time.time()
        camera_state['stop_event'].clear()
        camera_state['analyzer'] = BehaviorAnalyzer(segment_duration=10)

    analyzer = camera_state['analyzer']
    save_path = str(session_dirs['video'] / f"camera_{camera_id}_live.mp4")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        with camera_state['lock']:
            camera_state['running'] = False
        logger.error(f"无法打开摄像头: {camera_id}")
        return

    fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info("正在预热摄像头...")
    for _ in range(5):
        ret, _ = cap.read()
        if not ret:
            break
    logger.info("摄像头预热完成")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    frame_count = 0
    all_behaviors = []

    try:
        while not camera_state['stop_event'].is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("摄像头读取失败")
                time.sleep(0.01)
                continue

            frame_count += 1
            persons = detector.detect(frame)
            behaviors = []
            vis_frame = frame.copy()

            for person in persons:
                behavior, conf = detector.classify_behavior(person)
                behaviors.append({
                    'person_id': person.person_id,
                    'behavior': behavior,
                    'confidence': conf
                })
                vis_frame = detector._draw_detection(vis_frame, person, behavior, conf)

            elapsed = time.time() - camera_state['start_time']
            info_text = f"Frame: {frame_count} | Persons: {len(persons)} | Time: {elapsed:.0f}s"
            cv2.putText(vis_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            writer.write(vis_frame)

            analyzer.update(behaviors=behaviors, timestamp=datetime.now())
            all_behaviors.extend(behaviors)

            _, jpeg = cv2.imencode('.jpg', vis_frame)
            with camera_state['latest_frame_lock']:
                camera_state['latest_frame'] = jpeg.tobytes()

            with camera_state['lock']:
                camera_state['frame_count'] = frame_count

            if frame_count % 30 == 0:
                summary = analyzer.get_realtime_summary()
                logger.info(f"摄像头: {frame_count} 帧, "
                           f"检测 {summary['current_persons']} 人, "
                           f"已运行 {elapsed:.1f}秒")

    except Exception as e:
        logger.error(f"摄像头检测出错: {e}")
    finally:
        cap.release()
        writer.release()
        with camera_state['lock']:
            camera_state['running'] = False
        logger.info("摄像头已关闭，资源已释放")


import cv2


# ========== Flask路由 ==========

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/check_api_key', methods=['GET'])
def check_api_key():
    has_key = api_key_store['key'] is not None or bool(os.getenv('KIMI_API_KEY') or os.getenv('MOONSHOT_API_KEY'))
    return jsonify({'has_key': has_key})


@app.route('/api/set_api_key', methods=['POST'])
def set_api_key():
    data = request.get_json()
    key = data.get('api_key', '').strip()
    if not key:
        return jsonify({'success': False, 'message': 'API Key不能为空'}), 400

    try:
        test_client = KimiClient(api_key=key)
        api_key_store['key'] = key

        env_path = project_root / '.env'
        lines = []
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

        key_found = False
        new_lines = []
        for line in lines:
            if line.startswith('KIMI_API_KEY=') or line.startswith('MOONSHOT_API_KEY='):
                new_lines.append(f'KIMI_API_KEY={key}\n')
                key_found = True
            else:
                new_lines.append(line)
        if not key_found:
            new_lines.append(f'KIMI_API_KEY={key}\n')

        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        os.environ['KIMI_API_KEY'] = key
        os.environ['MOONSHOT_API_KEY'] = key

        return jsonify({'success': True, 'message': 'API Key设置成功并已验证'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'API Key验证失败: {str(e)}'}), 400


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if not (api_key_store['key'] or os.getenv('KIMI_API_KEY') or os.getenv('MOONSHOT_API_KEY')):
        return jsonify({'success': False, 'message': '请先设置API Key'}), 403

    if 'video' not in request.files:
        return jsonify({'success': False, 'message': '未找到视频文件'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择文件'}), 400

    ext = Path(file.filename).suffix
    if ext.lower() not in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
        return jsonify({'success': False, 'message': f'不支持的视频格式: {ext}'}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}{ext}"
    save_path = UPLOAD_FOLDER / filename
    file.save(str(save_path))

    video_name = Path(file.filename).stem
    session_dirs = create_session_dirs(video_name)

    with video_task_state['lock']:
        video_task_state['session_dirs'] = session_dirs
        video_task_state['progress'] = 'uploading'
        video_task_state['result'] = None

    thread = threading.Thread(
        target=video_detection_task,
        args=(str(save_path), session_dirs, video_name),
        daemon=True
    )
    thread.start()

    return jsonify({
        'success': True,
        'message': '视频已上传，开始检测',
        'session_dir': str(session_dirs['session'])
    })


@app.route('/api/video_progress', methods=['GET'])
def video_progress():
    with video_task_state['lock']:
        progress = video_task_state['progress']
        result = video_task_state['result']
    return jsonify({'progress': progress, 'result': result})


@app.route('/api/video_stream')
def video_stream():
    def generate():
        while True:
            with video_task_state['latest_frame_lock']:
                frame = video_task_state['latest_frame']
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            with video_task_state['lock']:
                if video_task_state['progress'] in ('done', 'error'):
                    break
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    if not (api_key_store['key'] or os.getenv('KIMI_API_KEY') or os.getenv('MOONSHOT_API_KEY')):
        return jsonify({'success': False, 'message': '请先设置API Key'}), 403

    with camera_state['lock']:
        if camera_state['running']:
            return jsonify({'success': False, 'message': '摄像头已在运行中'}), 400

    camera_id = request.get_json().get('camera_id', 0) if request.get_json() else 0
    session_dirs = create_session_dirs(f"camera_{camera_id}")

    with camera_state['lock']:
        camera_state['session_dirs'] = session_dirs

    thread = threading.Thread(
        target=camera_detection_task,
        args=(camera_id, session_dirs),
        daemon=True
    )
    thread.start()

    time.sleep(1)

    with camera_state['lock']:
        running = camera_state['running']

    if running:
        return jsonify({'success': True, 'message': '摄像头已开启'})
    else:
        return jsonify({'success': False, 'message': '摄像头开启失败，请检查设备'}), 500


@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    with camera_state['lock']:
        if not camera_state['running']:
            return jsonify({'success': False, 'message': '摄像头未在运行'}), 400
        camera_state['stop_event'].set()

    timeout = 10
    start_wait = time.time()
    while time.time() - start_wait < timeout:
        with camera_state['lock']:
            if not camera_state['running']:
                break
        time.sleep(0.1)

    analyzer = camera_state['analyzer']
    session_dirs = camera_state['session_dirs']
    frame_count = camera_state['frame_count']

    if analyzer is None or frame_count == 0:
        return jsonify({'success': False, 'message': '未检测到有效数据'}), 400

    stats = analyzer.get_statistics()
    behavior_counts = {
        name: data['count']
        for name, data in stats['behavior_distribution'].items()
    }
    actual_duration = time.time() - camera_state['start_time'] if camera_state['start_time'] else 0

    stats_data = {
        'total_frames': frame_count,
        'total_persons': stats['summary']['student_count'],
        'behaviors': behavior_counts,
        'focus_rate': stats['summary']['overall_focus_rate'],
        'duration_seconds': actual_duration,
        'raw_statistics': stats
    }

    video_name = f"camera_0"
    save_statistics(stats_data, session_dirs, video_name)

    kimi_result = do_kimi_analysis(stats_data, session_dirs)
    final_report = generate_final_report(stats_data, kimi_result, session_dirs, video_name)

    return jsonify({'success': True, 'result': final_report})


@app.route('/api/camera_status', methods=['GET'])
def camera_status():
    with camera_state['lock']:
        running = camera_state['running']
        frame_count = camera_state['frame_count']
    elapsed = 0
    if running and camera_state['start_time']:
        elapsed = time.time() - camera_state['start_time']
    return jsonify({'running': running, 'frame_count': frame_count, 'elapsed': round(elapsed, 1)})


@app.route('/api/camera_stream')
def camera_stream():
    def generate():
        while True:
            with camera_state['latest_frame_lock']:
                frame = camera_state['latest_frame']
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            with camera_state['lock']:
                if not camera_state['running']:
                    break
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/reports', methods=['GET'])
def list_reports():
    reports = []

    if not OUTPUT_DIR.exists():
        return jsonify({'reports': reports})

    for date_dir in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for session_dir in sorted(date_dir.iterdir(), reverse=True):
            report_file = session_dir / 'final_report.json'
            if not report_file.exists():
                continue
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                stat = report_file.stat()
                reports.append({
                    'id': str(session_dir.relative_to(OUTPUT_DIR)).replace('\\', '/'),
                    'name': session_dir.name,
                    'date': date_dir.name,
                    'path': str(session_dir),
                    'analysis_time': report_data.get('analysis_info', {}).get('analysis_time', ''),
                    'video_name': report_data.get('analysis_info', {}).get('video_name', ''),
                    'focus_rate': report_data.get('statistics', {}).get('focus_rate', 0),
                    'total_persons': report_data.get('statistics', {}).get('total_persons', 0),
                    'total_frames': report_data.get('statistics', {}).get('total_frames', 0),
                    'has_kimi': report_data.get('kimi_analysis') is not None,
                    'modified_time': stat.st_mtime
                })
            except Exception as e:
                logger.error(f"读取报告失败 {report_file}: {e}")

    reports.sort(key=lambda x: x.get('modified_time', 0), reverse=True)
    return jsonify({'reports': reports})


@app.route('/api/report/<path:report_id>', methods=['GET'])
def get_report(report_id):
    logger.info(f"获取报告请求: report_id={report_id}")
    report_path = OUTPUT_DIR / report_id / 'final_report.json'
    logger.info(f"查找报告路径: {report_path}")
    if not report_path.exists():
        logger.warning(f"报告不存在: {report_path}")
        return jsonify({'success': False, 'message': '报告不存在'}), 404
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        logger.info(f"报告读取成功: {report_path}")
        return jsonify({'success': True, 'report': report_data})
    except Exception as e:
        logger.error(f"读取报告失败: {e}")
        return jsonify({'success': False, 'message': f'读取报告失败: {str(e)}'}), 500


if __name__ == '__main__':
    logger.info("课堂行为智能分析系统 - Web应用启动")
    logger.info("访问 http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
