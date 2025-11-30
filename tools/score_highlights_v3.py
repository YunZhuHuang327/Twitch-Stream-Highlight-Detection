"""
Score highlight intensity - V3
Fixes: Strict title relevance detection + negative examples to prevent false positives
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from openai import OpenAI
import time
import re


def parse_timestamp(timestamp_str: str) -> float:
    """Convert HH:MM:SS to seconds"""
    parts = timestamp_str.split(':')
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_merged_events(json_path: str) -> List[Dict]:
    """Load merged events JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_time_windows(
    events: List[Dict],
    window_size: int = 30,
    stride: int = 15
) -> List[Dict]:
    """Create sliding time windows over events"""
    if not events:
        return []

    start_sec = events[0]['timestamp_sec']
    end_sec = events[-1]['timestamp_sec']

    windows = []
    current_start = start_sec

    while current_start < end_sec:
        current_end = current_start + window_size

        window_events = [
            e for e in events
            if current_start <= e['timestamp_sec'] < current_end
        ]

        if window_events:
            windows.append({
                'start_time': current_start,
                'end_time': current_end,
                'start_timestamp': format_timestamp(current_start),
                'end_timestamp': format_timestamp(current_end),
                'events': window_events
            })

        current_start += stride

    return windows


def format_window_for_prompt(window: Dict) -> str:
    """Format a time window into text for GPT prompt"""
    lines = []
    lines.append(f"Time Range: {window['start_timestamp']} - {window['end_timestamp']}")
    lines.append("")

    for event in window['events']:
        timestamp = event['timestamp']
        event_type = event['type']

        if event_type == 'ASR':
            text = event.get('text', '')
            lines.append(f"{timestamp} [ASR] {text}")

        elif event_type == 'CHAT_EVENT':
            label = event.get('event_label', '')
            text = event.get('text', '')
            if text:
                lines.append(f"{timestamp} [{label}] {text}")
            else:
                lines.append(f"{timestamp} [{label}]")

        elif event_type == 'VISUAL':
            description = event.get('description', '')
            lines.append(f"{timestamp} [VISUAL] {description}")

    return '\n'.join(lines)


def extract_title_entities(video_title: str) -> List[str]:
    """Extract key entities from video title"""
    entities = []

    # Extract @ mentions (collaborators)
    mentions = re.findall(r'@(\w+)', video_title)
    entities.extend(mentions)

    # Extract W/ or WITH mentions
    with_matches = re.findall(r'w[/\s]+(\w+)', video_title, re.IGNORECASE)
    entities.extend(with_matches)

    return [e.lower() for e in entities if len(e) > 2]


def score_window_highlights(
    client: OpenAI,
    window: Dict,
    video_title: str = "",
    title_entities: Optional[List[str]] = None,
    video_context: str = ""
) -> Dict:
    """
    Score highlight intensity for a time window using GPT-4o-mini
    V3: Strict title relevance detection + negative examples
    """
    window_text = format_window_for_prompt(window)

    # Build context section
    context_lines = []
    if video_title:
        context_lines.append(f"Video Title: {video_title}")
    if title_entities:
        entities_str = ', '.join(title_entities)
        context_lines.append(f"Title Keywords to Match: {entities_str}")

    context_section = "\n".join(context_lines) if context_lines else "No additional context provided"

    # Build title keywords pattern for checking
    title_keywords_note = ""
    if title_entities:
        title_keywords_note = f"\nIMPORTANT: Title mentions these people/places: {', '.join(title_entities)}"

    prompt = f"""You are an expert at identifying highlight moments in Twitch IRL/social livestreams.

{context_section}{title_keywords_note}

Analyze the following 30-second segment and score its "highlight intensity" from 0-10:

{window_text}

CRITICAL SCORING FACTORS (weighted):
- Video Title Context (40%): Does this segment relate to people/places/events in the title?
- Streamer Narrative & Content (35%): Engaging stories, conversations, reactions, reveals
- Chat Engagement Spikes (25%): Audience excitement and participation

SCORING GUIDELINES (use these ranges precisely):
- 0–2: Passive or filler content (walking, waiting, no meaningful interaction)
- 3–4: Minor interactions or setup (small talk, transitional moments)
- 5–6: Moderate interest or buildup (planning, anticipation, casual conversations)
- 7–8: Active event or emotional peak (engaging stories, reactions, introductions)
- 9–10: Key highlight moment (multi-factor convergence: title relevance + narrative + engagement)

TITLE RELEVANCE DETECTION (STRICT - must have explicit evidence):

To assign title_relevance > 0, you MUST find EXPLICIT evidence in ASR, CHAT, or VISUAL:

+3 (Directly interacting with title-mentioned person/place):
  ✅ ASR explicitly mentions the person's NAME ("Agent", "Agent00", "AE", collaborator name)
  ✅ CHAT shows NAME spam ("AGENT!!", "AE", repeated mentions of collaborator)
  ✅ VISUAL description mentions "two people" AND ASR/CHAT confirms collaborator name
  ✅ ASR mentions being AT the title-mentioned location ("in Little Italy", "at this restaurant")

+2 (Talking about or building up to title-mentioned person/event):
  ✅ ASR discusses meeting/plans with title-mentioned person by name
  ✅ Building anticipation ("can't wait to see [name]", "meeting [name] soon")
  ✅ Discussing the title-mentioned location/event

+1 (Tangentially related to title theme):
  ✅ ASR mentions title theme without specifics (food, exploring, TwitchCon context)
  ✅ General location context matching title

0 (No connection - DEFAULT):
  ❌ Generic fan interactions without name confirmation
  ❌ Random stranger interactions (even if chat is excited)
  ❌ No location/collaborator mentions in ASR/CHAT/VISUAL

CRITICAL WARNING - Common False Positives to AVOID:
❌ DO NOT assume every fan interaction is with title-mentioned collaborator
❌ DO NOT give +3 just because there's excitement or photo requests
❌ DO NOT assume chat spikes mean title-relevant person is present
❌ ALWAYS check ASR/CHAT for NAME CONFIRMATION before giving +2 or +3

IMPORTANT RULES:
✅ Title-relevant moments (with name confirmation) get priority:
   - First meeting with confirmed title-mentioned person: 8-10
   - Ongoing interactions with confirmed collaborator: 7-9
   - Discussing plans with named person: 6-8

✅ Narrative quality matters even without title relevance:
   - Engaging multi-turn conversations: 6-8
   - Emotional reactions or reveals: 7-9
   - Storytelling with buildup: 6-8

❌ Chat spikes WITHOUT substantial content = max 5-6:
   - Random emoji spam: 3-4
   - Brief reactions without context: 4-5

CONSISTENCY CHECK:
Before finalizing the score, compare against the previous 3 segments in your memory and keep scoring consistent across similar intensity levels.

RESPONSE FORMAT (strict JSON, no markdown, no extra text):

EXAMPLE 1 (Confirmed title-relevant interaction - HIGH SCORE):
{{
  "highlight_score": 9,
  "title_relevance": 3,
  "reasoning": "ASR explicitly mentions 'Agent' and 'Agent00' multiple times. CHAT shows 'AGENT!!' spam. VISUAL describes two people interacting. Clear first meeting moment with title-mentioned collaborator (+3).",
  "key_moments": [
    "00:15:31 ASR: 'Oh my God, Agent'",
    "00:15:45 CHAT: Multiple 'AGENT!!' messages",
    "00:16:00 Extended conversation with Agent00"
  ]
}}

EXAMPLE 2 (Strong narrative, NO title relevance - MEDIUM SCORE):
{{
  "highlight_score": 7,
  "title_relevance": 0,
  "reasoning": "Emily tells engaging story about zoo visit with Maya. Good ASR density and moderate chat engagement (LAUGH spikes). No mentions of Agent00 or Little Italy. Strong narrative but not title-related.",
  "key_moments": [
    "00:23:10 Story setup with suspense",
    "00:23:25 Punchline delivery",
    "00:23:30 Chat reactions (LAUGH)"
  ]
}}

EXAMPLE 3 (Chat spike only, weak content - LOW SCORE):
{{
  "highlight_score": 4,
  "title_relevance": 0,
  "reasoning": "High chat activity (SPIKE_HIGH) but ASR only shows filler content ('um', 'yeah', 'okay'). Brief emoji spam without substance. No title keywords mentioned.",
  "key_moments": [
    "00:45:12 Chat spike (emoji spam only)"
  ]
}}

EXAMPLE 4 (FALSE POSITIVE to AVOID - fan interaction WITHOUT name confirmation):
{{
  "highlight_score": 5,
  "title_relevance": 0,
  "reasoning": "Fan asks for photo and says 'I'm a big fan'. Chat spike present. BUT no name mentions in ASR/CHAT - cannot confirm if this is Agent00 or random fan. Generic interaction, no title connection.",
  "key_moments": [
    "01:03:20 Fan greeting and photo request"
  ]
}}

Now score this segment (JSON only, match the example format exactly):"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Validate required fields
        required_fields = ['highlight_score', 'title_relevance', 'reasoning', 'key_moments']
        for field in required_fields:
            if field not in result:
                if field == 'highlight_score':
                    result[field] = 0
                elif field == 'title_relevance':
                    result[field] = 0
                elif field == 'reasoning':
                    result[field] = 'Missing reasoning'
                elif field == 'key_moments':
                    result[field] = []

        # Ensure correct types and ranges
        result['highlight_score'] = max(0, min(10, int(result.get('highlight_score', 0))))
        result['title_relevance'] = max(0, min(3, int(result.get('title_relevance', 0))))

        return {
            'start_time': window['start_time'],
            'end_time': window['end_time'],
            'start_timestamp': window['start_timestamp'],
            'end_timestamp': window['end_timestamp'],
            'highlight_score': result['highlight_score'],
            'title_relevance': result['title_relevance'],
            'reasoning': result.get('reasoning', ''),
            'key_moments': result.get('key_moments', []),
            'tokens_used': response.usage.total_tokens
        }

    except Exception as e:
        print(f"  [ERROR] Failed to score window {window['start_timestamp']}: {e}")
        return {
            'start_time': window['start_time'],
            'end_time': window['end_time'],
            'start_timestamp': window['start_timestamp'],
            'end_timestamp': window['end_timestamp'],
            'highlight_score': 0,
            'title_relevance': 0,
            'reasoning': f'Error: {str(e)}',
            'key_moments': [],
            'tokens_used': 0
        }


def score_all_windows(
    windows: List[Dict],
    client: OpenAI,
    video_title: str = "",
    video_context: str = "",
    delay_seconds: float = 0.1
) -> List[Dict]:
    """Score all windows with rate limiting"""
    scored_windows = []
    total_tokens = 0

    title_entities = extract_title_entities(video_title) if video_title else None

    print(f"\n[SCORING] Processing {len(windows)} windows...")
    if video_title:
        print(f"[CONTEXT] Video Title: {video_title}")
        if title_entities:
            print(f"[CONTEXT] Will check for mentions of: {', '.join(title_entities)}")

    for i, window in enumerate(windows, 1):
        print(f"  [{i}/{len(windows)}] {window['start_timestamp']}-{window['end_timestamp']}", end='')

        result = score_window_highlights(client, window, video_title, title_entities, video_context)
        scored_windows.append(result)

        total_tokens += result.get('tokens_used', 0)
        score = result['highlight_score']
        title_rel = result.get('title_relevance', 0)
        print(f" → Score:{score}/10 Title:{title_rel}/3 [tokens:{result.get('tokens_used', 0)}]")

        if i < len(windows):
            time.sleep(delay_seconds)

    print(f"\n[SCORING] Complete! Total tokens: {total_tokens:,}")
    return scored_windows


def main():
    parser = argparse.ArgumentParser(
        description='Score highlights V3 (strict title relevance + negative examples)'
    )
    parser.add_argument('--merged_events', type=str, required=True)
    parser.add_argument('--output', type=str, help='Output path (default: _scored_v3.json)')
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--stride', type=int, default=15)
    parser.add_argument('--title', type=str, default='', help='Video title (CRITICAL)')
    parser.add_argument('--context', type=str, default='')
    parser.add_argument('--delay', type=float, default=0.1)

    args = parser.parse_args()

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    print(f"[LOAD] Loading merged events from {args.merged_events}")
    events = load_merged_events(args.merged_events)
    print(f"[LOAD] Loaded {len(events)} events")

    print(f"[WINDOWS] Creating windows (size={args.window_size}s, stride={args.stride}s)")
    windows = create_time_windows(events, args.window_size, args.stride)
    print(f"[WINDOWS] Created {len(windows)} windows")

    scored_windows = score_all_windows(
        windows, client,
        video_title=args.title,
        video_context=args.context,
        delay_seconds=args.delay
    )

    if args.output:
        output_path = args.output
    else:
        # Output to prompt-v3 folder
        output_path = Path('outputs/prompt-v3/highlight_scores_v3.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[SAVE] Saving to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scored_windows, f, indent=2, ensure_ascii=False)

    # Statistics
    scores = [w['highlight_score'] for w in scored_windows]
    title_rels = [w.get('title_relevance', 0) for w in scored_windows]

    print(f"\n[STATS] Complete!")
    print(f"  Total windows: {len(scored_windows)}")
    print(f"  Avg score: {sum(scores)/len(scores):.2f}")
    print(f"  Avg title relevance: {sum(title_rels)/len(title_rels):.2f}")
    print(f"  High-scoring (≥7): {len([s for s in scores if s >= 7])}")
    print(f"  Title-relevant (≥2): {len([t for t in title_rels if t >= 2])}")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    main()
