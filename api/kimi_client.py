"""
Kimi 2.5 API对接模块
使用OpenAI兼容接口调用Moonshot AI的Kimi大模型
支持单轮对话、多轮对话和行为分析专用接口
"""

import os
import json
import logging
from typing import List, Dict, Optional, Generator, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# 使用openai库调用Kimi API
from openai import OpenAI

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """消息数据类"""
    role: str  # 'system', 'user', 'assistant'
    content: str
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {'role': self.role, 'content': self.content}


@dataclass
class ChatResponse:
    """对话响应数据类"""
    content: str
    role: str
    model: str
    usage: Dict
    finish_reason: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'content': self.content,
            'role': self.role,
            'model': self.model,
            'usage': self.usage,
            'finish_reason': self.finish_reason,
            'timestamp': self.timestamp.isoformat()
        }


class KimiClient:
    """
    Kimi API客户端
    
    使用OpenAI兼容接口调用Kimi大模型，支持：
    - 单轮对话
    - 多轮对话（带上下文记忆）
    - 流式输出
    - 课堂行为分析专用接口
    """
    
    # Kimi API配置
    BASE_URL = "https://api.moonshot.cn/v1"
    DEFAULT_MODEL = "kimi-k2.5"
    
    # 可用模型列表
    AVAILABLE_MODELS = [
        "kimi-k2.5",      # Kimi 2.5 主模型
        "kimi-k2",        # Kimi 2.0
        "kimi-latest",    # 最新版本
    ]
    
    def __init__(self, api_key: str = None, model: str = None, 
                 temperature: float = 0.7, max_tokens: int = 4096):
        """
        初始化Kimi客户端
        
        Args:
            api_key: API密钥，默认从环境变量KIMI_API_KEY获取
            model: 模型名称，默认kimi-k2-5
            temperature: 温度参数，控制创造性（0-1）
            max_tokens: 最大生成token数
        """
        # 支持多种环境变量名
        self.api_key = api_key or os.getenv('KIMI_API_KEY') or os.getenv('MOONSHOT_API_KEY')
        if not self.api_key:
            raise ValueError(
                "请提供API密钥或设置环境变量 KIMI_API_KEY 或 MOONSHOT_API_KEY\n"
                "获取方式：https://platform.moonshot.cn/"
            )
        
        self.model = model or self.DEFAULT_MODEL
        
        # kimi-k2.5模型只支持temperature=1
        if self.model == "kimi-k2.5":
            self.temperature = 1.0
        else:
            self.temperature = temperature
            
        self.max_tokens = max_tokens
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL
        )
        
        # 对话历史（用于多轮对话）
        self.conversation_history: List[Message] = []
        
        logger.info(f"Kimi客户端初始化完成，模型: {self.model}")
    
    def chat(self, message: str, system_prompt: str = None, 
             keep_history: bool = False) -> ChatResponse:
        """
        单轮对话
        
        Args:
            message: 用户消息
            system_prompt: 系统提示词（可选）
            keep_history: 是否保留对话历史
            
        Returns:
            ChatResponse: 模型响应
        """
        # 构建消息列表
        messages = []
        
        # 添加系统提示
        if system_prompt:
            messages.append(Message('system', system_prompt))
        
        # 添加历史记录（如果需要）
        if keep_history and self.conversation_history:
            messages.extend(self.conversation_history)
        
        # 添加当前消息
        messages.append(Message('user', message))
        
        # 调用API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[m.to_dict() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # 解析响应
            result = ChatResponse(
                content=response.choices[0].message.content,
                role=response.choices[0].message.role,
                model=response.model,
                usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason,
                timestamp=datetime.now()
            )
            
            # 更新历史记录
            if keep_history:
                self.conversation_history.append(Message('user', message))
                self.conversation_history.append(
                    Message('assistant', result.content)
                )
            
            logger.info(f"对话完成，使用token: {result.usage['total_tokens']}")
            return result
            
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise
    
    def chat_stream(self, message: str, system_prompt: str = None) -> Generator[str, None, None]:
        """
        流式对话（逐字返回）
        
        Args:
            message: 用户消息
            system_prompt: 系统提示词（可选）
            
        Yields:
            str: 生成的文本片段
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': message})
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"流式API调用失败: {e}")
            raise
    
    def analyze_behavior(self, behavior_data: Dict) -> Dict:
        """
        课堂行为分析专用接口
        
        Args:
            behavior_data: 行为统计数据，格式如下：
                {
                    'total_frames': 1000,
                    'total_persons': 30,
                    'behaviors': {
                        'raise_hand': 50,
                        'head_down': 120,
                        'lying_on_desk': 30,
                        'good_posture': 800,
                        'discussing': 100
                    },
                    'focus_rate': 0.75,
                    'duration': 300  # 秒
                }
                
        Returns:
            Dict: 分析结果，包含教学建议
        """
        # 构建分析提示词
        system_prompt = """你是一位资深教育专家，擅长分析课堂行为数据并提供教学改进建议。
请基于提供的学生行为统计数据，分析课堂状态并给出具体的教学建议。

分析维度：
1. 课堂专注度评估
2. 学生参与度分析
3. 潜在问题识别
4. 教学改进建议

请以JSON格式返回分析结果，包含以下字段：
- summary: 整体评价（字符串）
- focus_level: 专注度等级（高/中/低）
- engagement_score: 参与度评分（0-100）
- issues: 发现的问题列表（数组）
- suggestions: 改进建议列表（数组）
- attention_students: 需要关注的学生数量（整数）"""

        # 格式化数据
        prompt = f"""请分析以下课堂行为数据：

【数据统计】
- 监测时长: {behavior_data.get('duration', 0)}秒
- 总帧数: {behavior_data.get('total_frames', 0)}
- 检测人次: {behavior_data.get('total_persons', 0)}
- 整体专注度: {behavior_data.get('focus_rate', 0):.1%}

【行为分布】
{json.dumps(behavior_data.get('behaviors', {}), ensure_ascii=False, indent=2)}

请提供详细的分析报告。"""

        try:
            response = self.chat(prompt, system_prompt=system_prompt)
            
            # 尝试解析JSON
            content = response.content
            
            # 提取JSON部分（如果模型返回了markdown代码块）
            if '```json' in content:
                json_str = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                json_str = content.split('```')[1].split('```')[0]
            else:
                json_str = content
            
            analysis = json.loads(json_str.strip())
            
            # 添加元数据
            analysis['_meta'] = {
                'model': response.model,
                'timestamp': response.timestamp.isoformat(),
                'tokens_used': response.usage['total_tokens']
            }
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败，返回原始文本: {e}")
            return {
                'summary': response.content,
                'focus_level': '未知',
                'engagement_score': 0,
                'issues': [],
                'suggestions': [],
                'attention_students': 0,
                '_meta': {
                    'parse_error': str(e),
                    'raw_response': response.content
                }
            }
        except Exception as e:
            logger.error(f"行为分析失败: {e}")
            raise
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        logger.info("对话历史已清空")
    
    def get_history(self) -> List[Dict]:
        """获取对话历史"""
        return [m.to_dict() for m in self.conversation_history]
    
    def set_model(self, model: str):
        """
        切换模型
        
        Args:
            model: 模型名称
        """
        if model not in self.AVAILABLE_MODELS:
            logger.warning(f"未知模型: {model}，使用默认模型")
            model = self.DEFAULT_MODEL
        
        self.model = model
        logger.info(f"已切换到模型: {model}")


# 便捷函数：快速创建客户端
def create_client(api_key: str = None) -> KimiClient:
    """
    快速创建Kimi客户端
    
    Args:
        api_key: API密钥，默认从环境变量获取
        
    Returns:
        KimiClient: 客户端实例
    """
    return KimiClient(api_key=api_key)