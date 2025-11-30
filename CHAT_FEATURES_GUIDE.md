# 聊天特徵提取指南

這個工具從 Twitch 聊天記錄中提取時間窗口特徵，用於分析觀眾反應和生成帶標註的 transcript。**專為 ExtraEmily 頻道優化**。

## 文件說明

- **工具**: [tools/extract_chat_features.py](tools/extract_chat_features.py)
- **輸入**: 轉換後的聊天記錄 JSON（使用 `convert_chat_format.py` 生成）
- **輸出**: 特徵 JSON + 帶標註的 Transcript JSON

---

## 🎯 功能概述

### 核心功能

1. **時間窗口分析**：將聊天記錄切分成固定時間窗口（默認 5 秒）
2. **多維度特徵提取**：從基礎統計到情緒分析
3. **事件分類**：自動識別 12 種聊天事件類型
4. **Transcript 生成**：創建帶特徵標註的時間軸
5. **ASR 整合**：可選整合語音轉錄數據

---

## 📊 特徵工程設計

### 1. 基礎層（訊息密度與結構）

| 特徵名 | 說明 | 用途 |
|--------|------|------|
| `msg_per_sec` | 平均每秒訊息數 | 活躍度指標 |
| `unique_users` | 獨立使用者數 | 參與度指標 |
| `avg_msg_len` | 平均訊息長度 | 內容複雜度 |
| `emoji_rate` | 含表情貼的訊息比例 | 情緒表達強度 |
| `caps_rate` | 大寫字母占比 | 強情緒指標 |
| `question_mark_rate` | 疑問符號比例 | 困惑/疑問度 |

### 2. 情緒層（觀眾反應）

基於 **ExtraEmily 頻道實際數據分析**：

| 特徵名 | 關鍵詞/模式 | 說明 |
|--------|-------------|------|
| `laugh_rate` | lol, lmao, icant, lul, kekw, haha, exemegalul | 笑聲反應比例 |
| `positive_rate` | yay, wooo, lets go, love, amazing, slay, iconic, ate, queen, poggers, pog, hype | 正面情緒比例 |
| `negative_rate` | cringe, wtf, yikes, rip, nooo, sad, oof, pain, monka | 負面情緒比例 |
| `confused_rate` | huh, what, why, how, ???, confused, erm | 困惑反應比例 |
| `emily_signature_rate` | yump, saj, agahi, jah, caught, saved, assemble | ExtraEmily 特色詞彙 |
| `excitement_rate` | omg, holy, insane, crazy, wild | 興奮時刻 |
| `spam_repeat_rate` | 重複字元（如 "HAHAHAHA"） | Spam 程度 |

### 3. 表情貼層（視覺情緒）

基於 **ExtraEmily 頻道最常用的 56 個表情貼**：

| 特徵名 | 表情貼 | 說明 |
|--------|--------|------|
| `hype_emote_rate` | TwitchConHYPE, PogChamp, exemClap, exemSturdy, DinoDance, exemFloss, exemWiggle | 高能量表情貼 |
| `laugh_emote_rate` | exemEGALUL, LUL, exemLUL, KEKW | 笑聲表情貼 |
| `love_emote_rate` | exemILY, exemLove, <3, exemHey | 愛心/溫馨表情貼 |
| `concern_emote_rate` | NotLikeThis, exemSAJ, monkaS | 擔心/緊張表情貼 |
| `eating_emote_rate` | exemEat, exemNugget | 吃東西表情貼（Emily 特色） |

### 4. 複雜度層（資訊密度）

| 特徵名 | 說明 | 計算方式 |
|--------|------|----------|
| `chat_entropy` | 詞彙多樣性 | Shannon entropy |
| `burstiness` | 訊息時間間隔變異 | 變異係數 (CV) |
| `clip_keyword_rate` | 剪輯關鍵詞頻率 | "clip", "clipit", "clipped" |
| `z_score_activity` | 相對活躍度 | 與全局均值比較 |

---

## 🏷️ 事件分類系統

### 12 種聊天事件類型

基於測試數據（89,112 條訊息，4,567 個 5 秒窗口）的分佈：

| 事件類型 | 頻率 | 判定邏輯 | 代表場景 |
|----------|------|----------|----------|
| `CHAT_NORMAL` | 52.6% | 無特殊反應 | 正常對話 |
| `CHAT_SPIKE_LAUGH` | 22.9% | laugh_rate > 20% 或 laugh_emote_rate > 30% | 爆笑時刻 |
| `CHAT_SPIKE_CONFUSED` | 10.5% | confused_rate > 15% 或 question_mark_rate > 20% | 困惑/看不懂 |
| `CHAT_SPIKE_HIGH` | 6.4% | msg_per_sec > 2.0 且 (laugh_rate > 15% 或 positive_rate > 10%) 且 高表情貼率 | 高能量爆發 |
| `CHAT_SPIKE_RELATIVE` | 5.8% | z_score > 2.0 | 相對活躍時段 |
| `CHAT_SPIKE_LOVE` | 4.3% | love_emote_rate > 15% 或 (positive_rate > 15% 且 emily_signature_rate > 5%) | 溫馨/感動時刻 |
| `CHAT_SPIKE_SPAM` | 4.1% | spam_repeat_rate > 30% | Spam/重複訊息 |
| `CHAT_CALM` | 2.9% | msg_per_sec < 0.5 | 平靜/冷場時段 |
| `CHAT_SPIKE_EATING` | 2.4% | eating_emote_rate > 10% | 吃東西時刻（Emily 特色） |
| `CHAT_SPIKE_CLIP_MOMENT` | 0.6% | clip_keyword_rate > 5% 且有笑聲或正面情緒 | 值得剪輯的精彩瞬間 |
| `CHAT_SPIKE_EXCITEMENT` | 0.4% | excitement_rate > 10% 且 caps_rate > 30% | 極度興奮 |
| `CHAT_SPIKE_CRINGE` | 0.1% | negative_rate > 10% 且 concern_emote_rate > 10% | 社死/尷尬 |

**注意**：一個窗口可能同時觸發多個事件標籤。

---

## 🚀 使用方法

### 1. 基本特徵提取

```bash
python tools/extract_chat_features.py \
    --chat_file "chat.json" \
    --output_features "chat_features.json" \
    --window_size 5
```

### 2. 生成帶標註的 Transcript

```bash
python tools/extract_chat_features.py \
    --chat_file "chat.json" \
    --output_features "chat_features.json" \
    --output_transcript "chat_transcript.json" \
    --window_size 5
```

### 3. 整合 ASR 數據

```bash
python tools/extract_chat_features.py \
    --chat_file "chat.json" \
    --output_features "chat_features.json" \
    --output_transcript "chat_transcript.json" \
    --asr_file "asr.json" \
    --window_size 5
```

### 4. 合併聊天特徵到 ASR（推薦用於訓練）

將聊天特徵直接嵌入 ASR 檔案，每個 ASR 記錄都包含對應時間的聊天特徵：

```bash
python tools/extract_chat_features.py \
    --chat_file "chat.json" \
    --asr_file "asr.txt" \
    --merge_into_asr \
    --output_asr "asr_with_chat.json" \
    --window_size 5
```

**支援格式**：
- ASR 可以是 `.txt` 格式（`HH:MM:SS: text`）或 `.json` 格式
- 輸出的 JSON 每個 ASR 記錄都包含完整的 `chat_features` 欄位

### 5. 生成可讀的 Transcript（ASR + 事件標籤）

生成人類可讀的文字格式，只包含 ASR 和聊天事件標籤：

```bash
python tools/extract_chat_features.py \
    --chat_file "chat.json" \
    --asr_file "asr.txt" \
    --readable_transcript \
    --output_readable "readable_transcript.txt" \
    --window_size 5
```

**輸出格式範例**：
```
00:00:00 [ASR] Empire State is this one
00:00:07 [ASR] Empire State Realty Trust
00:00:20 [CHAT_SPIKE_CLIP_MOMENT]
00:00:20 [ASR] Oh
00:00:22 [ASR] Beautiful beautiful. Hi
00:00:30 [CHAT_SPIKE_LOVE]
00:00:30 [CHAT_SPIKE_CLIP_MOMENT]
00:00:30 [ASR] Oh my! Who is that?
00:01:21 [CHAT_SPIKE_HIGH]
00:01:21 [ASR] Ahhhh!
```

### 參數說明

- `--chat_file`: 輸入聊天記錄（使用 `convert_chat_format.py` 轉換）
- `--output_features`: 輸出特徵 JSON 文件
- `--output_transcript`: 輸出 transcript JSON 文件（可選）
- `--asr_file`: ASR 數據文件（支援 `.txt` 或 `.json` 格式）
- `--merge_into_asr`: 將聊天特徵合併到 ASR 檔案中
- `--output_asr`: 輸出合併後的 ASR JSON 檔案
- `--readable_transcript`: 生成可讀的文字格式 transcript
- `--output_readable`: 輸出可讀 transcript 的檔案路徑
- `--window_size`: 時間窗口大小（秒），默認 5

---

## 📁 輸出格式

### 1. 合併的 ASR 檔案 (`asr_with_chat.json`) - 推薦用於訓練

每個 ASR 記錄都包含完整的聊天特徵：

```json
[
  {
    "start": 7.0,
    "end": 9.0,
    "text": "Empire State Realty Trust",
    "chat_features": {
      "msg_count": 3,
      "msg_per_sec": 0.6,
      "unique_users": 3,
      "avg_msg_len": 11.0,
      "emoji_rate": 0.0,
      "caps_rate": 0.606,
      "question_mark_rate": 0.0,
      "laugh_rate": 0.0,
      "positive_rate": 0.0,
      "negative_rate": 0.0,
      "confused_rate": 0.0,
      "emily_signature_rate": 0.333,
      "spam_repeat_rate": 0.333,
      "excitement_rate": 0.0,
      "hype_emote_rate": 0.0,
      "laugh_emote_rate": 0.0,
      "love_emote_rate": 0.0,
      "concern_emote_rate": 0.0,
      "eating_emote_rate": 0.0,
      "chat_entropy": 2.322,
      "burstiness": 0.0,
      "clip_keyword_rate": 0.0
    }
  },
  ...
]
```

### 2. 可讀的 Transcript (`readable_transcript.txt`)

純文字格式，適合人類閱讀和快速掃描：

```
00:00:00 [ASR] Empire State is this one
00:00:07 [ASR] Empire State Realty Trust
00:00:09 [ASR] Are you trying to find the Empire State?
00:00:20 [CHAT_SPIKE_CLIP_MOMENT]
00:00:20 [ASR] Oh
00:00:22 [ASR] Beautiful beautiful. Hi
00:00:30 [CHAT_SPIKE_LOVE]
00:00:30 [CHAT_SPIKE_CLIP_MOMENT]
00:00:30 [ASR] Oh my! Who is that?
00:01:21 [CHAT_SPIKE_HIGH]
00:01:21 [ASR] Ahhhh!
```

### 3. 特徵文件 (`chat_features.json`)

```json
{
  "0": {
    "msg_count": 6,
    "msg_per_sec": 1.2,
    "unique_users": 6,
    "avg_msg_len": 8.5,
    "emoji_rate": 0.0,
    "caps_rate": 0.45,
    "question_mark_rate": 0.0,
    "laugh_rate": 0.0,
    "positive_rate": 0.0,
    "negative_rate": 0.0,
    "confused_rate": 0.0,
    "emily_signature_rate": 0.83,
    "spam_repeat_rate": 0.0,
    "excitement_rate": 0.0,
    "hype_emote_rate": 0.0,
    "laugh_emote_rate": 0.0,
    "love_emote_rate": 0.0,
    "concern_emote_rate": 0.0,
    "eating_emote_rate": 0.0,
    "chat_entropy": 1.79,
    "burstiness": 0.47,
    "clip_keyword_rate": 0.0,
    "z_score_activity": -0.96,
    "events": ["CHAT_NORMAL"],
    "window_start": 5,
    "window_end": 10
  },
  ...
}
```

### 2. Transcript 文件 (`chat_transcript.json`)

```json
[
  {
    "timestamp": 80,
    "timestamp_str": "00:01:20",
    "window_end": 85,
    "events": ["CHAT_SPIKE_HIGH"],
    "metrics": {
      "msg_per_sec": 9.0,
      "unique_users": 45,
      "laugh_rate": 0.044,
      "positive_rate": 0.156,
      "emoji_rate": 0.311,
      "z_score": 1.82
    },
    "asr": [
      {
        "text": "Oh my god, this is insane!",
        "start": 78.5,
        "end": 81.2
      }
    ]
  },
  ...
]
```

---

## 📈 實際數據分析結果

基於 ExtraEmily 的 TwitchCon 直播（89,112 條訊息）：

### 整體統計
- **總訊息數**: 89,112
- **時間窗口數**: 4,567（5 秒/窗口）
- **總表情貼數**: 21,330

### 情緒分佈
- 笑聲反應: 15.0%
- 正面情緒: 3.2%
- 困惑反應: 5.0%
- 負面情緒: 1.1%

### 特色詞彙使用
- `caught`: 1.7%（最常見的 Emily 特色詞）
- `yump`: 1.2%
- `saj`: 0.9%

### 前 5 大表情貼
1. TwitchConHYPE: 5,124 次
2. exemEGALUL: 2,266 次
3. DinoDance: 1,257 次
4. exemClap: 1,045 次
5. LUL: 1,019 次

---

## 💡 應用場景

### 1. Highlight 檢測輔助
結合 `CHAT_SPIKE_HIGH` 和 `CHAT_SPIKE_CLIP_MOMENT` 事件自動標記可能的精彩時刻。

### 2. 觀眾情緒分析
追蹤情緒特徵隨時間的變化，分析直播效果。

### 3. 訓練數據增強
將聊天特徵作為額外輸入，提升 highlight 檢測模型性能。

### 4. 內容創作參考
分析哪些內容引發最多觀眾反應，優化未來直播策略。

---

## 🔧 進階自定義

### 修改關鍵詞和表情貼

編輯 [tools/extract_chat_features.py](tools/extract_chat_features.py) 的頂部定義：

```python
EMILY_KEYWORDS = {
    'signature': ['yump', 'saj', ...],  # 添加你的關鍵詞
    'laugh': ['lol', 'lmao', ...],
    ...
}

EMILY_EMOTES = {
    'hype': ['TwitchConHYPE', ...],  # 添加你的表情貼
    ...
}
```

### 調整事件閾值

修改 `classify_event()` 方法中的條件：

```python
# 例如：提高 CHAT_SPIKE_HIGH 的閾值
if (features['msg_per_sec'] > 3.0 and  # 從 2.0 改為 3.0
    ...):
    events.append('CHAT_SPIKE_HIGH')
```

### 更改窗口大小

根據需求調整：
- **更小的窗口**（如 3 秒）：更精細的時間分辨率
- **更大的窗口**（如 10 秒）：更平滑的趨勢分析

---

## 🔄 完整工作流程

### 從原始數據到特徵提取

```bash
# 1. 轉換聊天格式
python tools/convert_chat_format.py \
    --input_chat "raw_chat.json" \
    --emote_csv "emote_text.csv" \
    --output_chat "converted_chat.json"

# 2. 準備訓練數據（生成 ASR）
python tools/prepare_highlight_data.py \
    --video_path "video.mp4" \
    --chat_file "converted_chat.json" \
    --highlights_file "highlights.json" \
    --output_dir "dataset/highlights"

# 3. 提取聊天特徵
python tools/extract_chat_features.py \
    --chat_file "converted_chat.json" \
    --output_features "chat_features.json" \
    --output_transcript "chat_transcript.json" \
    --asr_file "dataset/highlights/video_name/asr.json" \
    --window_size 5
```

---

## 📚 相關文檔

- [CHAT_CONVERSION_GUIDE.md](CHAT_CONVERSION_GUIDE.md) - 聊天格式轉換
- [HIGHLIGHT_USAGE_GUIDE.md](HIGHLIGHT_USAGE_GUIDE.md) - Highlight 檢測訓練
- [prepare_highlight_data.py](tools/prepare_highlight_data.py) - 訓練數據準備

---

## ❓ 常見問題

### Q: 為什麼專門針對 ExtraEmily？

因為不同主播的觀眾文化差異很大。ExtraEmily 的觀眾有特色詞彙（yump, saj）和特定表情貼，通用的情緒分析無法捕捉這些特性。

### Q: 可以用於其他主播嗎？

可以！但需要：
1. 收集該主播的聊天數據
2. 分析常用詞彙和表情貼
3. 修改 `EMILY_KEYWORDS` 和 `EMILY_EMOTES` 定義

### Q: 5 秒窗口是否合適？

5 秒是平衡時間分辨率和統計穩定性的經驗值。可以根據具體需求調整：
- IRL 直播：5-10 秒
- 遊戲直播（激烈）：3-5 秒
- 聊天直播：10-15 秒

### Q: 如何處理長視頻？

工具會自動處理任意長度的視頻。對於 6-8 小時的長直播，建議先用 `prepare_highlight_data.py` 的分段模式切分。

---

## 🎓 技術細節

### Shannon Entropy 計算

用於衡量詞彙多樣性：

```
H = -Σ p(w) * log₂(p(w))
```

其中 p(w) 是詞彙 w 的頻率。熵越高，詞彙越多樣化。

### Burstiness 計算

用於衡量訊息時間分佈的不均勻程度：

```
Burstiness = σ(Δt) / μ(Δt)
```

其中 Δt 是相鄰訊息的時間間隔。越低表示越均勻。

### Z-score 活躍度

相對於全局平均的偏離程度：

```
z = (x - μ) / σ
```

z > 2 表示顯著高於平均水平。

---

## 🎉 範例輸出解讀

### 高能量時刻 (CHAT_SPIKE_HIGH)

```json
{
  "timestamp": 80,
  "events": ["CHAT_SPIKE_HIGH"],
  "metrics": {
    "msg_per_sec": 9.0,      // 極高的訊息密度
    "laugh_rate": 0.044,     // 4.4% 笑聲
    "positive_rate": 0.156,  // 15.6% 正面情緒
    "emoji_rate": 0.311,     // 31.1% 表情貼使用
    "z_score": 1.82          // 高於平均 1.82 個標準差
  }
}
```

**解讀**：這個 5 秒窗口中，每秒有 9 條訊息，大量使用表情貼，伴隨笑聲和正面情緒，是典型的精彩時刻。

### 正常時段 (CHAT_NORMAL)

```json
{
  "timestamp": 15,
  "events": ["CHAT_NORMAL"],
  "metrics": {
    "msg_per_sec": 2.2,      // 正常訊息密度
    "laugh_rate": 0.091,     // 9.1% 笑聲
    "positive_rate": 0.0,    // 無特別正面情緒
    "emoji_rate": 0.0,       // 無表情貼
    "z_score": -0.61         // 略低於平均
  }
}
```

**解讀**：普通的對話時段，沒有特別激烈的反應。

---

## 📖 實際使用範例

### 範例：處理 TwitchCon 直播片段

```bash
# 步驟 1: 生成可讀的 transcript 用於檢視
python tools/extract_chat_features.py \
    --chat_file "123.json" \
    --asr_file "dataset/highlights/123/asr.txt" \
    --readable_transcript \
    --output_readable "dataset/highlights/123/readable_transcript.txt" \
    --window_size 5

# 步驟 2: 生成訓練用的 ASR+Chat 數據
python tools/extract_chat_features.py \
    --chat_file "123.json" \
    --asr_file "dataset/highlights/123/asr.txt" \
    --merge_into_asr \
    --output_asr "dataset/highlights/123/asr_with_chat.json" \
    --window_size 5
```

**輸出結果**：
- 總訊息數: 89,112 條
- ASR 記錄: 8,579 條
- 可讀 transcript: 10,854 行（8,579 ASR + 2,275 事件標籤）
- 處理時間: ~30 秒

**偵測到的事件分佈**：
- CHAT_SPIKE_HIGH: 347 次
- CHAT_SPIKE_LAUGH: 156 次
- CHAT_SPIKE_LOVE: 89 次
- CHAT_SPIKE_CLIP_MOMENT: 45 次

---

## 🎯 最佳實踐

### 1. 訓練 Highlight 模型
- 使用 `--merge_into_asr` 模式生成訓練資料
- 將 `chat_features` 作為額外的輸入特徵餵給模型
- 特別關注 `CHAT_SPIKE_HIGH` 和 `CHAT_SPIKE_CLIP_MOMENT` 事件

### 2. 分析觀眾反應
- 使用 `--readable_transcript` 模式快速瀏覽
- 尋找連續出現的 `CHAT_SPIKE_*` 標籤，這些通常是精彩時刻
- 結合 ASR 文字理解內容脈絡

### 3. 調整窗口大小
- **IRL 直播**（如 TwitchCon）：5 秒 ✅
- **激烈遊戲**（如 FPS）：3 秒
- **聊天互動**：10 秒
- **長時間活動**：15 秒

### 4. 批次處理多個影片

```bash
# 創建批次處理腳本
for video_dir in dataset/highlights/*/; do
    video_name=$(basename "$video_dir")
    python tools/extract_chat_features.py \
        --chat_file "chats/${video_name}.json" \
        --asr_file "${video_dir}asr.txt" \
        --merge_into_asr \
        --output_asr "${video_dir}asr_with_chat.json" \
        --window_size 5
done
```

---

## 🔍 除錯和驗證

### 檢查事件標籤分佈

```bash
# 統計各種事件出現次數
grep "\[CHAT_SPIKE" readable_transcript.txt | sort | uniq -c | sort -rn
```

### 找出高能量時刻

```bash
# 找出所有 CHAT_SPIKE_HIGH 事件
grep -B 1 "\[CHAT_SPIKE_HIGH\]" readable_transcript.txt | grep ASR
```

### 驗證聊天特徵質量

```python
import json

# 載入合併的 ASR 檔案
with open('asr_with_chat.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 檢查高活躍度片段
high_activity = [
    item for item in data
    if item['chat_features']['msg_per_sec'] > 5.0
]

print(f"找到 {len(high_activity)} 個高活躍度片段")
for item in high_activity[:5]:
    print(f"{item['start']:.1f}s - {item['text']}")
    print(f"  訊息/秒: {item['chat_features']['msg_per_sec']:.1f}")
```

---

## 💡 進階技巧

### 1. 結合多個事件類型

有些精彩時刻會同時觸發多個事件：

```bash
# 找出同時有 HIGH 和 CLIP_MOMENT 的時段
grep -A 1 "CHAT_SPIKE_HIGH" readable_transcript.txt | grep "CHAT_SPIKE_CLIP_MOMENT"
```

### 2. 時間窗口滑動

如果想要更平滑的特徵，可以使用更小的窗口（如 2 秒）然後進行移動平均。

### 3. 自定義事件偵測器

根據你的需求添加新的事件類型：

```python
# 在 classify_event() 中添加
if (features['msg_per_sec'] > 10.0 and
    features['caps_rate'] > 0.5):
    events.append('CHAT_SPIKE_EXTREME')
```

---

## 📝 總結

這個聊天特徵提取工具提供了三種主要輸出：

1. **JSON 特徵檔**（`chat_features.json`）- 用於深度分析
2. **合併的 ASR 檔**（`asr_with_chat.json`）- 用於訓練模型 ⭐ **推薦**
3. **可讀 Transcript**（`readable_transcript.txt`）- 用於人工檢視 ⭐ **推薦**

最常見的使用流程：
1. 先生成 **可讀 transcript** 快速檢視數據質量
2. 再生成 **合併的 ASR** 用於訓練模型
3. 根據需要調整窗口大小和事件閾值

祝訓練順利！🎉
