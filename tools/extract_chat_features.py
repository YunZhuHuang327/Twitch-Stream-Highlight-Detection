"""
聊天特徵提取工具：從聊天記錄中提取時間窗口特徵，生成帶標註的 transcript

針對 ExtraEmily 頻道優化的特徵工程系統

使用方法:
    # 提取特徵並生成帶標註的 transcript
    python tools/extract_chat_features.py \
        --chat_file "chat.json" \
        --output_features "chat_features.json" \
        --output_transcript "chat_transcript.json" \
        --window_size 5 \
        --asr_file "asr.json"  # 可選，用於整合 ASR
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import re
import sys
import os
from datetime import timedelta

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')


# ==================== ExtraEmily 專屬關鍵詞和表情貼定義 ====================

# 基於數據分析的 ExtraEmily 頻道特色
EMILY_KEYWORDS = {
    'signature': ['yump', 'saj', 'agahi', 'jah', 'caught', 'saved', 'assemble'],
    'laugh': ['lol', 'lmao', 'icant', 'lul', 'kekw', 'haha', 'exemegalul'],
    'positive': ['yay', 'wooo', 'lets go', 'love', 'amazing', 'slay', 'iconic', 'ate', 'queen', 'poggers', 'pog', 'hype'],
    'negative': ['cringe', 'wtf', 'yikes', 'rip', 'nooo', 'sad', 'oof', 'pain', 'monka'],
    'confused': ['huh', 'what', 'why', 'how', '???', 'confused', 'erm'],
    'emote_text': ['oooo', 'hii', 'mhm'],
    'clip_moment': ['clip', 'clipit', 'clipped'],
    'excitement': ['omg', 'holy', 'insane', 'crazy', 'wild'],
}

# ExtraEmily 頻道的常用表情貼（基於實際數據）
EMILY_EMOTES = {
    'hype': ['TwitchConHYPE', 'PogChamp', 'exemClap', 'exemSturdy', 'DinoDance', 'exemFloss', 'exemWiggle'],
    'laugh': ['exemEGALUL', 'LUL', 'exemLUL', 'KEKW'],
    'love': ['exemILY', 'exemLove', '<3', 'exemHey'],
    'reaction': ['exemFlushed', 'exemNod', 'exemPlot', 'exemErm', 'exemSure', 'exemWhat'],
    'eating': ['exemEat', 'exemNugget'],
    'cute': ['exemSmile', 'danWave', ':)'],
    'concern': ['NotLikeThis', 'exemSAJ', 'monkaS'],
}


class ChatFeatureExtractor:
    """聊天特徵提取器"""

    def __init__(self, window_size=5):
        """
        Args:
            window_size: 時間窗口大小（秒），默認 5 秒
        """
        self.window_size = window_size
        self.keywords = EMILY_KEYWORDS
        self.emotes = EMILY_EMOTES

    def load_chat(self, chat_file):
        """
        載入聊天數據

        預期格式:
        [
            {
                "timestamp": 8,
                "user": "username",
                "message": "hello",
                "emotes": ["exemClap"]
            },
            ...
        ]
        """
        with open(chat_file, 'r', encoding='utf-8') as f:
            messages = json.load(f)

        # 確保按時間排序
        messages.sort(key=lambda x: x['timestamp'])

        print(f"✓ 載入 {len(messages)} 條聊天訊息")
        return messages

    def create_time_windows(self, messages):
        """
        將訊息分組到時間窗口中

        Returns:
            dict: {window_start: [messages]}
        """
        if not messages:
            return {}

        max_time = int(messages[-1]['timestamp']) + 1
        windows = defaultdict(list)

        for msg in messages:
            window_start = int(msg['timestamp'] // self.window_size) * self.window_size
            windows[window_start].append(msg)

        print(f"✓ 創建 {len(windows)} 個時間窗口 (每個 {self.window_size} 秒)")
        return dict(windows)

    def extract_basic_features(self, messages):
        """
        提取基礎特徵（訊息密度與結構）
        """
        if not messages:
            return {
                'msg_count': 0,
                'msg_per_sec': 0,
                'unique_users': 0,
                'avg_msg_len': 0,
                'emoji_rate': 0,
                'caps_rate': 0,
                'question_mark_rate': 0,
            }

        msg_count = len(messages)
        unique_users = len(set(msg['user'] for msg in messages))

        # 訊息長度
        msg_lengths = [len(msg['message']) for msg in messages]
        avg_msg_len = np.mean(msg_lengths) if msg_lengths else 0

        # 表情貼比例
        emoji_count = sum(1 for msg in messages if msg.get('emotes', []))
        emoji_rate = emoji_count / msg_count if msg_count > 0 else 0

        # 大寫比例（強情緒）
        total_chars = sum(len(msg['message']) for msg in messages)
        caps_chars = sum(sum(1 for c in msg['message'] if c.isupper()) for msg in messages)
        caps_rate = caps_chars / total_chars if total_chars > 0 else 0

        # 疑問符號比例
        question_count = sum(1 for msg in messages if '?' in msg['message'])
        question_mark_rate = question_count / msg_count if msg_count > 0 else 0

        return {
            'msg_count': msg_count,
            'msg_per_sec': msg_count / self.window_size,
            'unique_users': unique_users,
            'avg_msg_len': float(avg_msg_len),
            'emoji_rate': float(emoji_rate),
            'caps_rate': float(caps_rate),
            'question_mark_rate': float(question_mark_rate),
        }

    def extract_emotion_features(self, messages):
        """
        提取情緒特徵
        """
        if not messages:
            return {k: 0.0 for k in ['laugh_rate', 'positive_rate', 'negative_rate',
                                      'confused_rate', 'emily_signature_rate',
                                      'spam_repeat_rate', 'excitement_rate']}

        msg_count = len(messages)

        # 計算各類關鍵詞比例
        laugh_count = self._count_keywords(messages, self.keywords['laugh'])
        positive_count = self._count_keywords(messages, self.keywords['positive'])
        negative_count = self._count_keywords(messages, self.keywords['negative'])
        confused_count = self._count_keywords(messages, self.keywords['confused'])
        emily_sig_count = self._count_keywords(messages, self.keywords['signature'])
        excitement_count = self._count_keywords(messages, self.keywords['excitement'])

        # 重複字元（spam）
        spam_count = sum(1 for msg in messages if re.search(r'(.)\1{3,}', msg['message']))

        return {
            'laugh_rate': laugh_count / msg_count,
            'positive_rate': positive_count / msg_count,
            'negative_rate': negative_count / msg_count,
            'confused_rate': confused_count / msg_count,
            'emily_signature_rate': emily_sig_count / msg_count,
            'spam_repeat_rate': spam_count / msg_count,
            'excitement_rate': excitement_count / msg_count,
        }

    def extract_emote_features(self, messages):
        """
        提取表情貼特徵
        """
        if not messages:
            return {k: 0.0 for k in ['hype_emote_rate', 'laugh_emote_rate',
                                      'love_emote_rate', 'concern_emote_rate',
                                      'eating_emote_rate']}

        msg_count = len(messages)
        all_emotes = []
        for msg in messages:
            all_emotes.extend(msg.get('emotes', []))

        hype_count = sum(1 for e in all_emotes if e in self.emotes['hype'])
        laugh_count = sum(1 for e in all_emotes if e in self.emotes['laugh'])
        love_count = sum(1 for e in all_emotes if e in self.emotes['love'])
        concern_count = sum(1 for e in all_emotes if e in self.emotes['concern'])
        eating_count = sum(1 for e in all_emotes if e in self.emotes['eating'])

        return {
            'hype_emote_rate': hype_count / msg_count,
            'laugh_emote_rate': laugh_count / msg_count,
            'love_emote_rate': love_count / msg_count,
            'concern_emote_rate': concern_count / msg_count,
            'eating_emote_rate': eating_count / msg_count,
        }

    def extract_complexity_features(self, messages):
        """
        提取資訊密度特徵（複雜度）
        """
        if not messages:
            return {
                'chat_entropy': 0.0,
                'burstiness': 0.0,
                'clip_keyword_rate': 0.0,
            }

        # Shannon entropy（詞彙多樣性）
        words = []
        for msg in messages:
            words.extend(msg['message'].lower().split())

        word_counts = Counter(words)
        total_words = len(words)
        entropy = 0.0

        if total_words > 0:
            for count in word_counts.values():
                p = count / total_words
                if p > 0:
                    entropy -= p * np.log2(p)

        # Burstiness（訊息間時間間隔的變異係數）
        timestamps = [msg['timestamp'] for msg in messages]
        if len(timestamps) > 1:
            intervals = np.diff(sorted(timestamps))
            if len(intervals) > 0 and np.mean(intervals) > 0:
                burstiness = np.std(intervals) / np.mean(intervals)
            else:
                burstiness = 0.0
        else:
            burstiness = 0.0

        # Clip 關鍵詞密度
        clip_count = self._count_keywords(messages, self.keywords['clip_moment'])
        clip_rate = clip_count / len(messages)

        return {
            'chat_entropy': float(entropy),
            'burstiness': float(burstiness),
            'clip_keyword_rate': float(clip_rate),
        }

    def _count_keywords(self, messages, keywords):
        """計算包含特定關鍵詞的訊息數"""
        count = 0
        for msg in messages:
            text = msg['message'].lower()
            if any(kw in text for kw in keywords):
                count += 1
        return count

    def classify_event(self, features, global_stats):
        """
        根據特徵分類事件類型

        Returns:
            list: 事件標籤列表（可能有多個）
        """
        events = []

        # 計算 z-score
        z_score = 0.0
        if global_stats['msg_per_sec_std'] > 0:
            z_score = (features['msg_per_sec'] - global_stats['msg_per_sec_mean']) / global_stats['msg_per_sec_std']

        # 1. CHAT_SPIKE_HIGH - 高能量爆發（爆笑/勝利）
        if (features['msg_per_sec'] > 2.0 and
            (features['laugh_rate'] > 0.15 or features['positive_rate'] > 0.1) and
            (features['hype_emote_rate'] > 0.2 or features['laugh_emote_rate'] > 0.2)):
            events.append('CHAT_SPIKE_HIGH')

        # 2. CHAT_SPIKE_LAUGH - 純粹爆笑時刻
        if (features['laugh_rate'] > 0.2 or features['laugh_emote_rate'] > 0.3):
            events.append('CHAT_SPIKE_LAUGH')

        # 3. CHAT_SPIKE_CONFUSED - 困惑反應
        if (features['confused_rate'] > 0.15 or
            (features['question_mark_rate'] > 0.2 and features['msg_per_sec'] > 1.0)):
            events.append('CHAT_SPIKE_CONFUSED')

        # 4. CHAT_SPIKE_CRINGE - 社死/尷尬
        if (features['negative_rate'] > 0.1 and features['concern_emote_rate'] > 0.1):
            events.append('CHAT_SPIKE_CRINGE')

        # 5. CHAT_SPIKE_LOVE - 溫馨/感動時刻
        if (features['love_emote_rate'] > 0.15 or
            (features['positive_rate'] > 0.15 and features['emily_signature_rate'] > 0.05)):
            events.append('CHAT_SPIKE_LOVE')

        # 6. CHAT_SPIKE_EATING - 吃東西時刻（ExtraEmily 特色）
        if features['eating_emote_rate'] > 0.1:
            events.append('CHAT_SPIKE_EATING')

        # 7. CHAT_SPIKE_EXCITEMENT - 興奮時刻
        if (features['excitement_rate'] > 0.1 and features['caps_rate'] > 0.3):
            events.append('CHAT_SPIKE_EXCITEMENT')

        # 8. CHAT_SPIKE_CLIP_MOMENT - 值得剪輯的瞬間
        if (features['clip_keyword_rate'] > 0.05 and
            (features['laugh_rate'] > 0.1 or features['positive_rate'] > 0.1)):
            events.append('CHAT_SPIKE_CLIP_MOMENT')

        # 9. CHAT_SPIKE_SPAM - Spam/重複訊息
        if features['spam_repeat_rate'] > 0.3:
            events.append('CHAT_SPIKE_SPAM')

        # 10. CHAT_SPIKE_RELATIVE - 相對於平均活躍
        if z_score > 2.0:
            events.append('CHAT_SPIKE_RELATIVE')

        # 11. CHAT_CALM - 平靜時段
        if features['msg_per_sec'] < 0.5:
            events.append('CHAT_CALM')

        # 儲存 z-score
        features['z_score_activity'] = float(z_score)

        return events if events else ['CHAT_NORMAL']

    def calculate_global_stats(self, all_features):
        """計算全局統計數據（用於 z-score）"""
        msg_per_secs = [f['msg_per_sec'] for f in all_features.values()]

        return {
            'msg_per_sec_mean': np.mean(msg_per_secs) if msg_per_secs else 0,
            'msg_per_sec_std': np.std(msg_per_secs) if msg_per_secs else 1,
        }

    def extract_all_features(self, messages):
        """
        提取所有特徵

        Returns:
            dict: {
                window_start: {
                    'basic': {...},
                    'emotion': {...},
                    'emote': {...},
                    'complexity': {...},
                    'events': [...]
                }
            }
        """
        print("\n🔍 開始提取特徵...")

        # 創建時間窗口
        windows = self.create_time_windows(messages)

        # 提取每個窗口的特徵
        all_features = {}
        for window_start, window_msgs in windows.items():
            basic = self.extract_basic_features(window_msgs)
            emotion = self.extract_emotion_features(window_msgs)
            emote = self.extract_emote_features(window_msgs)
            complexity = self.extract_complexity_features(window_msgs)

            # 合併所有特徵
            features = {**basic, **emotion, **emote, **complexity}
            all_features[window_start] = features

        # 計算全局統計
        global_stats = self.calculate_global_stats(all_features)

        # 分類事件
        for window_start, features in all_features.items():
            events = self.classify_event(features, global_stats)
            features['events'] = events
            features['window_start'] = window_start
            features['window_end'] = window_start + self.window_size

        print(f"✓ 提取完成！共 {len(all_features)} 個窗口")

        # 統計事件分佈
        event_counter = Counter()
        for features in all_features.values():
            event_counter.update(features['events'])

        print("\n📊 事件分佈:")
        for event, count in event_counter.most_common():
            print(f"   {event}: {count} ({count/len(all_features)*100:.1f}%)")

        return all_features


def format_timestamp(seconds):
    """將秒數轉換為 HH:MM:SS 格式"""
    td = timedelta(seconds=int(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def create_enriched_transcript(features, asr_data=None):
    """
    創建帶特徵標註的 transcript

    Args:
        features: 提取的特徵字典
        asr_data: ASR 數據（可選）

    Returns:
        list: 帶標註的 transcript
    """
    print("\n📝 創建帶標註的 transcript...")

    transcript = []

    for window_start, feature in sorted(features.items()):
        # 基本信息
        entry = {
            'timestamp': window_start,
            'timestamp_str': format_timestamp(window_start),
            'window_end': feature['window_end'],
            'events': feature['events'],
            'metrics': {
                'msg_per_sec': round(feature['msg_per_sec'], 2),
                'unique_users': feature['unique_users'],
                'laugh_rate': round(feature['laugh_rate'], 3),
                'positive_rate': round(feature['positive_rate'], 3),
                'emoji_rate': round(feature['emoji_rate'], 3),
                'z_score': round(feature['z_score_activity'], 2),
            }
        }

        # 如果有 ASR 數據，嘗試匹配
        if asr_data:
            matching_asr = []
            for asr_item in asr_data:
                if 'start' in asr_item and 'end' in asr_item:
                    # ASR 片段與窗口有重疊
                    if (asr_item['start'] <= feature['window_end'] and
                        asr_item['end'] >= window_start):
                        matching_asr.append({
                            'text': asr_item.get('text', ''),
                            'start': asr_item['start'],
                            'end': asr_item['end']
                        })

            if matching_asr:
                entry['asr'] = matching_asr

        transcript.append(entry)

    print(f"✓ 創建了 {len(transcript)} 條 transcript 記錄")

    return transcript


def load_asr_from_txt(asr_txt_file):
    """
    從 .txt 格式的 ASR 檔案載入數據

    格式: HH:MM:SS: text

    Returns:
        list: ASR 數據列表，格式為 [{'start': float, 'end': float, 'text': str}]
    """
    print(f"\n📖 載入 ASR 文本檔: {asr_txt_file}")

    asr_data = []

    with open(asr_txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # 解析格式: HH:MM:SS: text
        if ':' in line:
            parts = line.split(':', 3)  # 最多分割3次 (HH:MM:SS:text)
            if len(parts) >= 4:
                try:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = int(parts[2])
                    text = parts[3].strip()

                    # 轉換為秒數
                    start_time = hours * 3600 + minutes * 60 + seconds

                    # end 時間設為下一條記錄的開始時間，如果是最後一條則加5秒
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if ':' in next_line:
                            next_parts = next_line.split(':', 3)
                            if len(next_parts) >= 3:
                                next_hours = int(next_parts[0])
                                next_minutes = int(next_parts[1])
                                next_seconds = int(next_parts[2])
                                end_time = next_hours * 3600 + next_minutes * 60 + next_seconds
                            else:
                                end_time = start_time + 5
                        else:
                            end_time = start_time + 5
                    else:
                        end_time = start_time + 5

                    asr_data.append({
                        'start': float(start_time),
                        'end': float(end_time),
                        'text': text
                    })

                except (ValueError, IndexError) as e:
                    print(f"⚠️  警告: 無法解析第 {i+1} 行: {line}")
                    continue

    print(f"   ✓ 載入 {len(asr_data)} 條 ASR 記錄")
    return asr_data


def merge_chat_features_into_asr(asr_data, messages, extractor):
    """
    將聊天特徵直接合併到 ASR 記錄中

    Args:
        asr_data: ASR 數據列表
        messages: 聊天訊息列表
        extractor: ChatFeatureExtractor 實例

    Returns:
        list: 帶有聊天特徵的 ASR 記錄
    """
    print("\n🔄 將聊天特徵合併到 ASR 記錄中...")

    enriched_asr = []

    for i, asr_item in enumerate(asr_data):
        # 複製原始 ASR 記錄
        enriched_item = asr_item.copy()

        # 獲取 ASR 片段的時間範圍
        start_time = asr_item.get('start', 0)
        end_time = asr_item.get('end', start_time + extractor.window_size)

        # 獲取這個時間窗口內的訊息
        window_messages = [msg for msg in messages
                          if start_time <= msg['timestamp'] < end_time]

        # 提取特徵
        basic = extractor.extract_basic_features(window_messages)
        emotion = extractor.extract_emotion_features(window_messages)
        emote = extractor.extract_emote_features(window_messages)
        complexity = extractor.extract_complexity_features(window_messages)

        # 合併特徵
        chat_features = {**basic, **emotion, **emote, **complexity}

        # 添加到 ASR 記錄中
        enriched_item['chat_features'] = chat_features

        enriched_asr.append(enriched_item)

        # 進度顯示
        if (i + 1) % 100 == 0:
            print(f"   處理進度: {i+1}/{len(asr_data)}")

    print(f"✓ 完成處理 {len(enriched_asr)} 條 ASR 記錄")

    return enriched_asr


def create_readable_transcript(asr_data, messages, extractor, global_stats):
    """
    創建可讀的文字格式 transcript，只包含 ASR 和事件標籤

    格式:
    00:14:05 [ASR] Emily is talking
    00:14:07 [CHAT_SPIKE_HIGH]
    00:14:07 [ASR] Emily is laughing so hard
    00:14:10 [CHAT_SPIKE_LAUGH]
    00:14:10 [ASR] That was so funny

    Args:
        asr_data: ASR 數據列表
        messages: 聊天訊息列表
        extractor: ChatFeatureExtractor 實例
        global_stats: 全局統計數據

    Returns:
        str: 可讀的 transcript 文字
    """
    print("\n📝 創建可讀的 transcript...")

    # 計算每個時間窗口的事件標籤
    window_events = {}
    for window_start in range(0, int(max(msg['timestamp'] for msg in messages)) + extractor.window_size, extractor.window_size):
        window_end = window_start + extractor.window_size
        window_messages = [msg for msg in messages
                          if window_start <= msg['timestamp'] < window_end]

        if window_messages:
            basic = extractor.extract_basic_features(window_messages)
            emotion = extractor.extract_emotion_features(window_messages)
            emote = extractor.extract_emote_features(window_messages)
            complexity = extractor.extract_complexity_features(window_messages)

            features = {**basic, **emotion, **emote, **complexity}
            events = extractor.classify_event(features, global_stats)

            # 只保留非 NORMAL 事件
            if events and events != ['CHAT_NORMAL']:
                window_events[window_start] = events

    # 生成文字：只包含 ASR 和事件標籤
    lines = []
    last_window = -1

    for asr in asr_data:
        timestamp = asr['start']
        time_str = format_timestamp(timestamp)

        # 檢查是否進入新的時間窗口，如果有事件則插入標籤
        current_window = int(timestamp // extractor.window_size) * extractor.window_size
        if current_window != last_window and current_window in window_events:
            for event in window_events[current_window]:
                lines.append(f"{time_str} [{event}]")
            last_window = current_window

        # 添加 ASR
        lines.append(f"{time_str} [ASR] {asr['text']}")

    transcript_text = '\n'.join(lines)

    print(f"✓ 創建了 {len(lines)} 行 transcript")

    return transcript_text


def main():
    parser = argparse.ArgumentParser(
        description="從聊天記錄中提取時間窗口特徵（針對 ExtraEmily 頻道優化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 基本特徵提取:
   python tools/extract_chat_features.py \\
       --chat_file "chat.json" \\
       --output_features "chat_features.json" \\
       --window_size 5

2. 生成帶 ASR 的 transcript:
   python tools/extract_chat_features.py \\
       --chat_file "chat.json" \\
       --output_features "chat_features.json" \\
       --output_transcript "chat_transcript.json" \\
       --asr_file "asr.json" \\
       --window_size 5

3. 將聊天特徵直接合併到 ASR 檔案中（推薦用於訓練）:
   python tools/extract_chat_features.py \\
       --chat_file "chat.json" \\
       --asr_file "asr.json" \\
       --merge_into_asr \\
       --output_asr "asr_with_chat.json" \\
       --window_size 5
        """
    )

    parser.add_argument('--chat_file', required=True,
                       help="聊天記錄 JSON 文件")
    parser.add_argument('--output_features',
                       help="輸出特徵 JSON 文件（可選）")
    parser.add_argument('--output_transcript',
                       help="輸出帶標註的 transcript JSON 文件（可選）")
    parser.add_argument('--asr_file',
                       help="ASR 數據 JSON 文件（可選）")
    parser.add_argument('--merge_into_asr', action='store_true',
                       help="將聊天特徵直接合併到 ASR 檔案中（需要 --asr_file）")
    parser.add_argument('--output_asr',
                       help="輸出合併後的 ASR 檔案（與 --merge_into_asr 一起使用）")
    parser.add_argument('--readable_transcript', action='store_true',
                       help="生成可讀的混合 transcript（ASR + Chat + 事件標籤）")
    parser.add_argument('--output_readable',
                       help="輸出可讀 transcript 的檔案路徑")
    parser.add_argument('--window_size', type=int, default=5,
                       help="時間窗口大小（秒），默認 5")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🎯 ExtraEmily 聊天特徵提取器")
    print("="*60)

    # 創建提取器
    extractor = ChatFeatureExtractor(window_size=args.window_size)

    # 載入聊天數據
    messages = extractor.load_chat(args.chat_file)

    # 模式 1: 生成可讀的混合 transcript
    if args.readable_transcript:
        if not args.asr_file:
            print("❌ 錯誤: --readable_transcript 需要指定 --asr_file")
            return
        if not args.output_readable:
            print("❌ 錯誤: --readable_transcript 需要指定 --output_readable")
            return

        # 載入 ASR 數據
        if args.asr_file.endswith('.txt'):
            asr_data = load_asr_from_txt(args.asr_file)
        else:
            print(f"\n📖 載入 ASR 檔案: {args.asr_file}")
            with open(args.asr_file, 'r', encoding='utf-8') as f:
                asr_data = json.load(f)
            print(f"   ✓ 載入 {len(asr_data)} 條 ASR 記錄")

        # 計算全局統計（用於事件分類）
        print("\n📊 計算全局統計...")
        windows = extractor.create_time_windows(messages)
        all_features = {}
        for window_start, window_msgs in windows.items():
            basic = extractor.extract_basic_features(window_msgs)
            emotion = extractor.extract_emotion_features(window_msgs)
            emote = extractor.extract_emote_features(window_msgs)
            complexity = extractor.extract_complexity_features(window_msgs)
            all_features[window_start] = {**basic, **emotion, **emote, **complexity}
        global_stats = extractor.calculate_global_stats(all_features)

        # 生成可讀的 transcript
        transcript_text = create_readable_transcript(asr_data, messages, extractor, global_stats)

        # 保存
        output_path = Path(args.output_readable)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)

        print(f"\n✅ 可讀 transcript 已保存到: {output_path}")

        # 顯示預覽
        print("\n📋 前 50 行預覽:")
        print("=" * 60)
        for line in transcript_text.split('\n')[:50]:
            print(line)
        print("=" * 60)

    # 模式 2: 合併到 ASR 檔案
    elif args.merge_into_asr:
        if not args.asr_file:
            print("❌ 錯誤: --merge_into_asr 需要指定 --asr_file")
            return
        if not args.output_asr:
            print("❌ 錯誤: --merge_into_asr 需要指定 --output_asr")
            return

        # 載入 ASR 數據（支援 .json 和 .txt 格式）
        if args.asr_file.endswith('.txt'):
            asr_data = load_asr_from_txt(args.asr_file)
        else:
            print(f"\n📖 載入 ASR 檔案: {args.asr_file}")
            with open(args.asr_file, 'r', encoding='utf-8') as f:
                asr_data = json.load(f)
            print(f"   ✓ 載入 {len(asr_data)} 條 ASR 記錄")

        # 合併聊天特徵到 ASR
        enriched_asr = merge_chat_features_into_asr(asr_data, messages, extractor)

        # 保存結果
        output_path = Path(args.output_asr)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_asr, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 合併後的 ASR 已保存到: {output_path}")

        # 顯示範例
        if enriched_asr:
            print("\n📋 範例記錄:")
            item = enriched_asr[0]
            print(f"   - 時間: {item.get('start', 0):.2f} - {item.get('end', 0):.2f} 秒")
            print(f"   - ASR 文本: {item.get('text', '')[:50]}...")
            if 'chat_features' in item:
                print(f"   - 聊天訊息數: {item['chat_features']['msg_count']}")
                print(f"   - 訊息/秒: {item['chat_features']['msg_per_sec']:.2f}")
                print(f"   - 獨特用戶: {item['chat_features']['unique_users']}")

    # 模式 3: 傳統特徵提取
    else:
        # 提取特徵
        features = extractor.extract_all_features(messages)

        # 保存特徵
        if args.output_features:
            output_path = Path(args.output_features)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=2, ensure_ascii=False)

            print(f"\n✅ 特徵已保存到: {output_path}")

        # 如果需要生成 transcript
        if args.output_transcript:
            # 載入 ASR（如果有）
            asr_data = None
            if args.asr_file:
                with open(args.asr_file, 'r', encoding='utf-8') as f:
                    asr_data = json.load(f)
                print(f"✓ 載入 ASR 數據: {len(asr_data)} 條記錄")

            # 創建 transcript
            transcript = create_enriched_transcript(features, asr_data)

            # 保存 transcript
            transcript_path = Path(args.output_transcript)
            transcript_path.parent.mkdir(parents=True, exist_ok=True)

            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)

            print(f"✅ Transcript 已保存到: {transcript_path}")

    print("\n" + "="*60)
    print("🎉 完成！")
    print("="*60)


if __name__ == '__main__':
    main()
