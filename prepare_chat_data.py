"""
示例：如何准备聊天室数据用于继续训练

假设你有聊天室数据，需要转换成与 VidChapters 相同的格式
"""

# 示例数据结构
example_chat_data = {
    "video_id": "chat_session_001",
    "asr": """
00:00:10 用户A: 欢迎来到今天的讨论
00:00:15 用户B: 今天我们要讨论什么主题？
00:00:20 用户A: 我们来谈谈机器学习
00:02:30 用户C: 深度学习很有趣
00:05:00 用户A: 让我们总结一下今天的内容
""",
    "chapters": [
        {"timestamp": "00:00:00", "title": "开场与介绍"},
        {"timestamp": "00:00:20", "title": "机器学习讨论"},
        {"timestamp": "00:02:30", "title": "深度学习话题"},
        {"timestamp": "00:05:00", "title": "总结"}
    ],
    "duration": 360  # 6 分钟
}

# 步骤 1: 将聊天记录转换为 ASR 格式
def convert_chat_to_asr(chat_messages):
    """
    将聊天消息转换为 ASR 格式
    chat_messages: list of {"timestamp": "HH:MM:SS", "user": str, "message": str}
    """
    asr_lines = []
    for msg in chat_messages:
        asr_lines.append(f"{msg['timestamp']} {msg['user']}: {msg['message']}")
    return "\n".join(asr_lines) + "\n"

# 步骤 2: 定义章节
def create_chapters(chat_messages, chapter_boundaries):
    """
    根据对话内容创建章节
    chapter_boundaries: list of {"timestamp": str, "title": str}
    """
    return chapter_boundaries

# 步骤 3: 保存为 JSON 格式
import json
from pathlib import Path

def save_chat_data_for_training(output_dir, chat_data):
    """
    保存聊天数据供训练使用
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 ASR
    asr_file = output_dir / "asrs_chat.json"
    asr_dict = {
        chat_data["video_id"]: chat_data["asr"]
    }
    with open(asr_file, 'w', encoding='utf-8') as f:
        json.dump(asr_dict, f, ensure_ascii=False, indent=2)
    
    # 保存 Chapters
    chapters_file = output_dir / "chapters_chat.json"
    chapters_dict = {
        chat_data["video_id"]: {
            "duration": chat_data["duration"],
            "chapters": chat_data["chapters"]
        }
    }
    with open(chapters_file, 'w', encoding='utf-8') as f:
        json.dump(chapters_dict, f, ensure_ascii=False, indent=2)
    
    # 保存视频 ID 列表
    ids_file = output_dir / "chat_data.json"
    with open(ids_file, 'w', encoding='utf-8') as f:
        json.dump([chat_data["video_id"]], f, ensure_ascii=False, indent=2)
    
    print(f"Chat data saved to {output_dir}")
    return output_dir

# 使用示例
if __name__ == "__main__":
    # 准备你的聊天数据
    my_chat_data = {
        "video_id": "chat_001",
        "asr": convert_chat_to_asr([
            {"timestamp": "00:00:00", "user": "UserA", "message": "Hello everyone"},
            {"timestamp": "00:00:05", "user": "UserB", "message": "Hi there"},
            # ... 更多消息
        ]),
        "chapters": [
            {"timestamp": "00:00:00", "title": "Introduction"},
            # ... 更多章节
        ],
        "duration": 600
    }
    
    # 保存数据
    save_chat_data_for_training("dataset/docs/chat_data/", my_chat_data)
