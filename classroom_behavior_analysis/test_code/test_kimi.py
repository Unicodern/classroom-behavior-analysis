"""
Kimi API模块测试脚本
测试单轮对话、多轮对话、流式输出和行为分析功能
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv  # 导入库

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.kimi_client import KimiClient, create_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
api_key=os.environ.get("MOONSHOT_API_KEY")

def test_single_turn():
    """测试单轮对话"""
    print("\n" + "="*50)
    print("测试1: 单轮对话")
    print("="*50)
    
    try:
        # 创建客户端
        client = create_client(api_key)
        
        # 发送消息
        message = "请用一句话介绍Kimi大模型的特点。"
        print(f"\n用户: {message}")
        
        response = client.chat(message)
        
        print(f"\nKimi: {response.content}")
        print(f"\n[使用统计]")
        print(f"  模型: {response.model}")
        print(f"  Token: {response.usage['total_tokens']} "
              f"(提示: {response.usage['prompt_tokens']}, "
              f"生成: {response.usage['completion_tokens']})")
        
        return True
        
    except Exception as e:
        logger.error(f"单轮对话测试失败: {e}")
        return False


def test_multi_turn():
    """测试多轮对话（带上下文记忆）"""
    print("\n" + "="*50)
    print("测试2: 多轮对话（上下文记忆）")
    print("="*50)
    
    try:
        client = create_client(api_key)
        
        # 第一轮
        msg1 = "我最喜欢的颜色是蓝色。"
        print(f"\n用户: {msg1}")
        
        resp1 = client.chat(msg1, keep_history=True)
        print(f"Kimi: {resp1.content}")
        
        # 第二轮（应该记住上下文）
        msg2 = "我刚才说的最喜欢的颜色是什么？"
        print(f"\n用户: {msg2}")
        
        resp2 = client.chat(msg2, keep_history=True)
        print(f"Kimi: {resp2.content}")
        
        # 查看历史记录
        print(f"\n[对话历史] 共 {len(client.get_history())} 条消息")
        
        return True
        
    except Exception as e:
        logger.error(f"多轮对话测试失败: {e}")
        return False


def test_stream():
    """测试流式输出"""
    print("\n" + "="*50)
    print("测试3: 流式输出")
    print("="*50)
    
    try:
        client = create_client(api_key)
        
        message = "请写一首关于春天的短诗。"
        print(f"\n用户: {message}")
        print(f"\nKimi: ", end='', flush=True)
        
        # 流式接收
        full_response = ""
        for chunk in client.chat_stream(message):
            print(chunk, end='', flush=True)
            full_response += chunk
        
        print()  # 换行
        
        return True
        
    except Exception as e:
        logger.error(f"流式输出测试失败: {e}")
        return False


def test_behavior_analysis():
    """测试课堂行为分析"""
    print("\n" + "="*50)
    print("测试4: 课堂行为分析")
    print("="*50)
    
    try:
        client = create_client(api_key)
        
        # 模拟行为数据
        behavior_data = {
            'total_frames': 900,  # 30秒 @ 30fps
            'total_persons': 25,
            'behaviors': {
                'raise_hand': 15,
                'head_down': 180,
                'lying_on_desk': 45,
                'good_posture': 600,
                'discussing': 60
            },
            'focus_rate': 0.72,
            'duration': 30
        }
        
        print("\n[输入数据]")
        print(json.dumps(behavior_data, ensure_ascii=False, indent=2))
        
        print("\n[分析结果]")
        result = client.analyze_behavior(behavior_data)
        
        # 打印分析结果
        print(f"\n整体评价: {result.get('summary', 'N/A')}")
        print(f"专注度等级: {result.get('focus_level', 'N/A')}")
        print(f"参与度评分: {result.get('engagement_score', 0)}")
        
        if result.get('issues'):
            print(f"\n发现问题:")
            for issue in result['issues']:
                print(f"  - {issue}")
        
        if result.get('suggestions'):
            print(f"\n改进建议:")
            for suggestion in result['suggestions']:
                print(f"  - {suggestion}")
        
        print(f"\n需关注学生: {result.get('attention_students', 0)}人")
        
        # 打印元数据
        if '_meta' in result:
            print(f"\n[元数据]")
            print(f"  模型: {result['_meta'].get('model', 'N/A')}")
            print(f"  Token: {result['_meta'].get('tokens_used', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"行为分析测试失败: {e}")
        return False


def test_interactive():
    """交互式对话测试"""
    print("\n" + "="*50)
    print("测试5: 交互式对话（输入'quit'退出）")
    print("="*50)
    
    try:
        client = create_client(api_key)
        
        print("\n提示: 这是一个带上下文记忆的对话，Kimi会记住之前的对话内容。")
        print("特殊命令: 'quit'=退出, 'clear'=清空历史, 'history'=查看历史\n")
        
        while True:
            # 获取用户输入
            user_input = input("用户: ").strip()
            
            # 处理特殊命令
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            if user_input.lower() == 'clear':
                client.clear_history()
                print("[历史已清空]\n")
                continue
            
            if user_input.lower() == 'history':
                history = client.get_history()
                print(f"\n[历史记录] 共 {len(history)} 条:")
                for i, msg in enumerate(history, 1):
                    content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                    print(f"  {i}. [{msg['role']}] {content}")
                print()
                continue
            
            if not user_input:
                continue
            
            # 发送消息并获取响应
            try:
                print("Kimi: ", end='', flush=True)
                
                # 使用流式输出
                full_response = ""
                for chunk in client.chat_stream(user_input):
                    print(chunk, end='', flush=True)
                    full_response += chunk
                
                print()  # 换行
                
                # 保存到历史
                client.conversation_history.append({'role': 'user', 'content': user_input})
                client.conversation_history.append({'role': 'assistant', 'content': full_response})
                
            except Exception as e:
                print(f"\n[错误] {e}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
        return True
    except Exception as e:
        logger.error(f"交互式测试失败: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Kimi API模块测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行所有测试
  python test_kimi.py --all
  
  # 仅测试单轮对话
  python test_kimi.py --single
  
  # 仅测试多轮对话
  python test_kimi.py --multi
  
  # 测试流式输出
  python test_kimi.py --stream
  
  # 测试行为分析
  python test_kimi.py --analyze
  
  # 交互式对话
  python test_kimi.py --interactive
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='运行所有测试')
    parser.add_argument('--single', action='store_true',
                       help='测试单轮对话')
    parser.add_argument('--multi', action='store_true',
                       help='测试多轮对话')
    parser.add_argument('--stream', action='store_true',
                       help='测试流式输出')
    parser.add_argument('--analyze', action='store_true',
                       help='测试行为分析')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='交互式对话')
    
    args = parser.parse_args()
    
    # 检查API密钥
    if not os.getenv('MOONSHOT_API_KEY'):
        print("\n[警告] 未设置MOONSHOT_API_KEY环境变量")
        print("请设置环境变量: set MOONSHOT_API_KEY=your_api_key")
        print("或在代码中直接传入api_key参数\n")
        return
    
    # 如果没有指定任何测试，显示帮助
    if not any([args.all, args.single, args.multi, args.stream, 
                args.analyze, args.interactive]):
        parser.print_help()
        return
    
    # 运行测试
    results = []
    
    if args.all or args.single:
        results.append(("单轮对话", test_single_turn()))
    
    if args.all or args.multi:
        results.append(("多轮对话", test_multi_turn()))
    
    if args.all or args.stream:
        results.append(("流式输出", test_stream()))
    
    if args.all or args.analyze:
        results.append(("行为分析", test_behavior_analysis()))
    
    if args.interactive:
        test_interactive()
        return
    
    # 打印测试总结
    if results:
        print("\n" + "="*50)
        print("测试总结")
        print("="*50)
        
        for name, success in results:
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{name}: {status}")
        
        total = len(results)
        passed = sum(1 for _, s in results if s)
        print(f"\n总计: {passed}/{total} 通过")


if __name__ == '__main__':
    main()