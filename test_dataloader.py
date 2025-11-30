import sys
sys.path.append('.')
from src.data.highlight_data import HighlightData
from pathlib import Path

print('🔍 测试数据加载器...\n')

# 加载数据
data = HighlightData('dataset/highlights', subset='train')

# 测试 video_id
video_id = 'v1'
print(f'Video ID: {video_id}')

# 测试 ASR
asr = data.get_asr(video_id)
print(f'ASR 长度: {len(asr)} 字符')
print(f'ASR 前 100 字: {asr[:100]}...\n')

# 测试 highlights
highlights = data.load_highlights(video_id)
print(f'Highlights 数量: {len(highlights)}')
for hl in highlights:
    print(f'  - [{hl["start_time_str"]}-{hl["end_time_str"]}] {hl["type"]}: {hl["description"][:40]}')

# 测试 chat
chat_data = data.load_chat_data(video_id)
print(f'\n聊天消息数: {chat_data["total_messages"]}')

# 测试 chat summary
chat_summary = data.get_chat_summary(video_id)
print(f'\n聊天摘要 ({len(chat_summary)} 字符):\n{chat_summary[:200]}...\n')

print('✅ 数据加载器测试通过！')
