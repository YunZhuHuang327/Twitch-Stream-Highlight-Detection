# Highlight 标注模板和指南

## 📋 简化标注格式（推荐）

只需要标注 3 个字段：
- `start_time`: 开始时间
- `end_time`: 结束时间  
- `type`: highlight 类型

### 标注模板文件: `highlight_template.json`

```json
[
  {
    "start_time": "00:15:30",
    "end_time": "00:18:45",
    "type": "exciting_moment"
  },
  {
    "start_time": "00:45:20",
    "end_time": "00:47:30",
    "type": "funny_moment"
  }
]
```

## 🏷️ Highlight 类型说明

### 1. `exciting_moment` - 激动时刻
- 游戏中的精彩操作
- 赢得比赛/关键胜利
- 完成困难挑战
- 获得稀有物品

**示例**:
- 吃鸡/五杀
- Boss 首杀
- 完美连击
- 关键逆转

### 2. `funny_moment` - 搞笑时刻
- 游戏中的失误/BUG
- 意外的搞笑情况
- 主播的幽默互动
- 观众的有趣评论

**示例**:
- 角色卡墙穿模
- 低级失误
- 搞笑对话
- 意外死亡

### 3. `skill_showcase` - 技术展示
- 高难度操作
- 完美执行技巧
- 精准预判
- 专业解说

**示例**:
- 连续爆头
- 完美走位
- 技术教学
- 高端操作

### 4. `emotional_moment` - 情感时刻
- 感人的瞬间
- 主播的情绪表达
- 与观众的深度互动
- 回忆/感谢时刻

**示例**:
- 感谢观众支持
- 讲述个人故事
- 达成重要里程碑
- 情感共鸣

### 5. `chat_peak` - 聊天高峰
- 聊天室爆炸时刻
- 大量弹幕/礼物
- 观众强烈反应
- 话题热度高涨

**示例**:
- 刷屏时刻
- 礼物雨
- 全员参与讨论
- 梗图/表情包刷屏

## ⏱️ 时间格式

### 支持两种格式:

1. **字符串格式** (推荐):
   ```json
   "start_time": "01:23:45"  // HH:MM:SS
   "start_time": "23:45"      // MM:SS
   ```

2. **秒数格式**:
   ```json
   "start_time": 5025  // 5025 秒 = 01:23:45
   ```

### 快速转换:
- 1 小时 = 3600 秒
- 30 分钟 = 1800 秒
- 5 分钟 = 300 秒

## 📝 标注工作流程

### 方法 1: 手工标注（精确但慢）

1. 播放视频，发现 highlight 时暂停
2. 记录开始时间
3. 播放到 highlight 结束，记录结束时间
4. 判断类型（exciting/funny/skill/emotional/chat_peak）
5. 添加到 JSON 文件

**预计时间**: 6 小时视频 → 1-2 小时标注

### 方法 2: 辅助工具标注（推荐）

使用视频播放器的标记功能:
1. VLC/PotPlayer 可以添加书签
2. 标记所有 highlight 位置
3. 导出时间戳
4. 转换为 JSON 格式

**预计时间**: 6 小时视频 → 30-60 分钟标注

### 方法 3: 聊天数据辅助（最快）

利用聊天高峰找 highlight:
1. 查看 `chat.json` 中的 `peak_moments`
2. 这些时刻通常对应 highlight
3. 只需验证和分类

**预计时间**: 6 小时视频 → 15-30 分钟标注

## 🎯 标注质量建议

### 时间精度:
- ✅ 好: highlight 开始前 1-2 秒，结束后 1-2 秒
- ⚠️ 可以: highlight 开始前 5 秒，结束后 5 秒
- ❌ 不好: 包含大量无关内容

### Highlight 长度:
- 🎯 最佳: 10 秒 - 2 分钟
- ⚠️ 可以: 5 秒 - 5 分钟
- ❌ 避免: < 5 秒（太短）或 > 10 分钟（太长）

### 数量分布:
对于 6 小时视频，建议标注:
- **最少**: 10-15 个 highlight（每 30 分钟 1 个）
- **推荐**: 20-30 个 highlight（每 15 分钟 1-2 个）
- **最多**: 40-60 个 highlight（每 5-10 分钟 1 个）

### 类型平衡:
尽量保持多样性:
```
exciting_moment: 40%
funny_moment: 25%
skill_showcase: 20%
emotional_moment: 10%
chat_peak: 5%
```

## 💡 标注技巧

### 1. 先粗后精
- 第一遍: 快速浏览，只标记大致位置
- 第二遍: 精确调整时间

### 2. 利用视频剪辑软件
- Premiere/DaVinci Resolve 可以添加标记
- 直接在时间轴上标记 in/out 点
- 导出 EDL 或 XML，转换为 JSON

### 3. 批量标注
- 一次标注多个视频
- 建立标注规范
- 保持一致性

### 4. 参考聊天数据
```python
# 查看聊天高峰时刻
import json
with open('chat.json') as f:
    chat = json.load(f)
    
for peak in chat['peak_moments']:
    print(f"时间: {peak['timestamp_str']}, 强度: {peak['intensity']}")
    print(f"关键词: {peak['keywords']}")
```

## 🔧 验证工具

创建一个简单的验证脚本:

```python
import json

def validate_highlights(file_path):
    with open(file_path) as f:
        highlights = json.load(f)
    
    valid_types = ['exciting_moment', 'funny_moment', 'skill_showcase', 
                   'emotional_moment', 'chat_peak']
    
    for i, hl in enumerate(highlights):
        # 检查必需字段
        assert 'start_time' in hl, f"Highlight {i}: 缺少 start_time"
        assert 'end_time' in hl, f"Highlight {i}: 缺少 end_time"
        assert 'type' in hl, f"Highlight {i}: 缺少 type"
        
        # 检查类型有效性
        assert hl['type'] in valid_types, f"Highlight {i}: 无效的 type '{hl['type']}'"
        
        # TODO: 检查时间有效性
    
    print(f"✅ 验证通过! 共 {len(highlights)} 个 highlights")

validate_highlights('highlights.json')
```

## 📦 批量标注模板生成

```python
def generate_template(video_count=10):
    """生成多个视频的空白标注模板"""
    for i in range(1, video_count + 1):
        template = {
            "video_id": f"stream_{i:03d}",
            "highlights": [
                {
                    "start_time": "00:00:00",
                    "end_time": "00:00:00",
                    "type": "exciting_moment",
                    "note": "标注说明"
                }
            ]
        }
        
        with open(f'highlights_stream_{i:03d}.json', 'w', encoding='utf-8') as f:
            json.dump(template['highlights'], f, indent=2, ensure_ascii=False)

generate_template(10)
```

## 🎓 学习曲线

- **第 1 个视频**: 2-3 小时（学习规范）
- **第 2-5 个视频**: 1-2 小时（熟练操作）
- **第 6-10 个视频**: 45-90 分钟（建立节奏）
- **第 10+ 个视频**: 30-60 分钟（高效标注）

---

**记住**: 标注质量比数量更重要！宁可少标注几个视频，也要保证每个 highlight 都准确。
