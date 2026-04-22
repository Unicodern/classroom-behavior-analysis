import os
from openai import OpenAI
from dotenv import load_dotenv  # 导入库

load_dotenv()
api_key=os.environ.get("MOONSHOT_API_KEY")

client = OpenAI(
    api_key=api_key, # 使用你创建的api密钥
    base_url="https://api.moonshot.cn/v1",
)

try:
    response = client.chat.completions.create(
        model="kimi-k2.5",  #使用免费的大模型
        messages=[
            {
                "role": "user",
                "content": "你好，介绍下你自己吧",
            }
        ],
        stream=False,
    )

    # 响应内容访问方式
    print(response.choices[0].message.content)

except Exception as e:
    print(f"调用API时出错: {e}")
    import traceback

    traceback.print_exc()
