"""
Score highlight intensity for merged events using GPT-4o-mini - V2
Based on groundtruth analysis with data-driven weights and strict JSON format
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from openai import OpenAI
import time


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
    """
    Create sliding time windows over events

    Args:
        events: List of events sorted by timestamp
        window_size: Window size in seconds (default 30s)
        stride: How much to move window forward (default 15s for 50% overlap)

    Returns:
        List of windows, each containing:
        - start_time: Window start time in seconds
        - end_time: Window end time in seconds
        - events: Events in this window
    """
    if not events:
        return []

    start_sec = events[0]['timestamp_sec']
    end_sec = events[-1]['timestamp_sec']

    windows = []
    current_start = start_sec

    while current_start < end_sec:
        current_end = current_start + window_size

        # Get events in this window
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
    """
    Format a time window into text for GPT prompt

    Args:
        window: Dict with start_time, end_time, events

    Returns:
        Formatted string with all events in the window
    """
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
    """
    Extract key entities from video title (collaborators, locations, events)
    Simple keyword extraction - can be enhanced with NER later
    """
    # Common keywords to look for
    import re

    entities = []

    # Extract @ mentions (collaborators)
    mentions = re.findall(r'@(\w+)', video_title)
    entities.extend(mentions)

    # Extract W/ or WITH mentions
    with_matches = re.findall(r'w[/\s]+(\w+)', video_title, re.IGNORECASE)
    entities.extend(with_matches)

    # Common location/event patterns (can be expanded)
    # For now, return extracted mentions
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
    V2: Data-driven weights + strict JSON format + examples

    Returns:
        Dict with:
        - start_time, end_time
        - highlight_score: 0-10 integer
        - reasoning: Why this score
        - key_moments: List of specific timestamps that are interesting
        - title_relevance: 0-3 integer
    """
    window_text = format_window_for_prompt(window)

    # Build context section
    context_lines = []
    if video_title:
        context_lines.append(f"Video Title: {video_title}")
    if title_entities:
        context_lines.append(f"Title Keywords: {', '.join(title_entities)}")
    if video_context:
        context_lines.append(f"Additional Context: {video_context}")

    context_section = "\n".join(context_lines) if context_lines else "No additional context provided"

    prompt = f"""You are an expert at identifying highlight moments in Twitch IRL/social livestreams.

{context_section}

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

TITLE RELEVANCE SCORING (compute separately, then add as bonus):
- 0: No connection to title keywords or themes
- +1: Tangentially related (mentions title keywords in passing)
- +2: Talking about or building up to title-mentioned person/event
- +3: Directly interacting with title-mentioned guest or at title-mentioned location

IMPORTANT RULES:
✅ Title-relevant moments get priority:
   - First meeting/interaction with title-mentioned person: auto 8-10
   - Ongoing interactions with them: 7-9
   - Discussing plans related to title theme: 6-8

✅ Narrative quality matters even without chat:
   - Engaging multi-turn conversations: 6-8
   - Emotional reactions or reveals: 7-9
   - Storytelling with buildup: 6-8

❌ Chat spikes WITHOUT substance = max 5-6:
   - Random emoji spam: 3-4
   - Brief reactions without context: 4-5

CONSISTENCY CHECK:
Before finalizing the score, compare against the previous 3 segments in your memory and keep scoring consistent across similar intensity levels.

RESPONSE FORMAT (strict JSON, no markdown, no extra text):

EXAMPLE 1 (Title-relevant interaction):
{{
  "highlight_score": 9,
  "title_relevance": 3,
  "reasoning": "First meeting with Agent00 (mentioned in title). Multiple chat spikes and Emily's excited reactions. Clear title relevance (+3 bonus).",
  "key_moments": [
    "00:15:31 Emily recognizes Agent00",
    "00:15:45 Enthusiastic greeting conversation",
    "00:16:00 Chat celebrates the meetup"
  ]
}}

EXAMPLE 2 (Strong narrative, no title relevance):
{{
  "highlight_score": 7,
  "title_relevance": 0,
  "reasoning": "Emily tells engaging story about trip planning. Good ASR density (0.4/s) and moderate chat engagement. No direct title connection.",
  "key_moments": [
    "00:23:10 Story setup with suspense",
    "00:23:25 Punchline delivery",
    "00:23:30 Chat reactions (LAUGH)"
  ]
}}

EXAMPLE 3 (Chat spike only, weak content):
{{
  "highlight_score": 4,
  "title_relevance": 0,
  "reasoning": "High chat activity (SPIKE_HIGH) but only brief emoji spam. ASR shows filler content ('um', 'yeah'). Low substance.",
  "key_moments": [
    "00:45:12 Chat spike (emoji spam)"
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
            max_tokens=350,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Validate required fields
        required_fields = ['highlight_score', 'title_relevance', 'reasoning', 'key_moments']
        for field in required_fields:
            if field not in result:
                print(f"  [WARN] Missing field '{field}', using default")
                if field == 'highlight_score':
                    result[field] = 0
                elif field == 'title_relevance':
                    result[field] = 0
                elif field == 'reasoning':
                    result[field] = 'Missing reasoning'
                elif field == 'key_moments':
                    result[field] = []

        # Ensure correct types
        result['highlight_score'] = int(result.get('highlight_score', 0))
        result['title_relevance'] = int(result.get('title_relevance', 0))

        # Clip values to valid ranges
        result['highlight_score'] = max(0, min(10, result['highlight_score']))
        result['title_relevance'] = max(0, min(3, result['title_relevance']))

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
    """
    Score all windows with rate limiting

    Args:
        windows: List of time windows
        client: OpenAI client
        video_title: Title of the video
        video_context: Additional context about the video
        delay_seconds: Delay between API calls to avoid rate limits

    Returns:
        List of scored windows
    """
    scored_windows = []
    total_tokens = 0

    # Extract title entities once
    title_entities = extract_title_entities(video_title) if video_title else None

    print(f"\n[SCORING] Processing {len(windows)} windows...")
    if video_title:
        print(f"[CONTEXT] Video Title: {video_title}")
        if title_entities:
            print(f"[CONTEXT] Title Keywords: {', '.join(title_entities)}")

    for i, window in enumerate(windows, 1):
        print(f"  [{i}/{len(windows)}] Scoring {window['start_timestamp']} - {window['end_timestamp']}", end='')

        result = score_window_highlights(client, window, video_title, title_entities, video_context)
        scored_windows.append(result)

        total_tokens += result.get('tokens_used', 0)
        score = result['highlight_score']
        title_rel = result.get('title_relevance', 0)
        print(f" → Score: {score}/10 (title:{title_rel}) [tokens: {result.get('tokens_used', 0)}]")

        # Rate limiting
        if i < len(windows):
            time.sleep(delay_seconds)

    print(f"\n[SCORING] Complete! Total tokens used: {total_tokens:,}")
    return scored_windows


def main():
    parser = argparse.ArgumentParser(
        description='Score highlight intensity using GPT-4o-mini V2 (data-driven weights)'
    )
    parser.add_argument(
        '--merged_events',
        type=str,
        required=True,
        help='Path to merged events JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for scored windows JSON (default: same dir as input with _scored_v2.json suffix)'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=30,
        help='Window size in seconds (default: 30)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=15,
        help='Stride in seconds (default: 15 for 50%% overlap)'
    )
    parser.add_argument(
        '--title',
        type=str,
        default='',
        help='Video title (CRITICAL: extracts collaborators/locations for context)'
    )
    parser.add_argument(
        '--context',
        type=str,
        default='',
        help='Additional video context/description'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.1,
        help='Delay between API calls in seconds (default: 0.1)'
    )

    args = parser.parse_args()

    # Load API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    # Load events
    print(f"[LOAD] Loading merged events from {args.merged_events}")
    events = load_merged_events(args.merged_events)
    print(f"[LOAD] Loaded {len(events)} events")

    # Create windows
    print(f"[WINDOWS] Creating windows (size={args.window_size}s, stride={args.stride}s)")
    windows = create_time_windows(events, args.window_size, args.stride)
    print(f"[WINDOWS] Created {len(windows)} windows")

    # Score windows
    scored_windows = score_all_windows(
        windows,
        client,
        video_title=args.title,
        video_context=args.context,
        delay_seconds=args.delay
    )

    # Save results
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.merged_events)
        output_path = input_path.parent / f"{input_path.stem}_scored_v2.json"

    print(f"\n[SAVE] Saving scored windows to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scored_windows, f, indent=2, ensure_ascii=False)

    # Print statistics
    scores = [w['highlight_score'] for w in scored_windows]
    title_relevances = [w.get('title_relevance', 0) for w in scored_windows]

    avg_score = sum(scores) / len(scores) if scores else 0
    avg_title_rel = sum(title_relevances) / len(title_relevances) if title_relevances else 0

    high_score_windows = [w for w in scored_windows if w['highlight_score'] >= 7]
    title_relevant_windows = [w for w in scored_windows if w.get('title_relevance', 0) >= 2]

    print(f"\n[STATS] Scoring complete!")
    print(f"  Total windows: {len(scored_windows)}")
    print(f"  Average score: {avg_score:.2f}")
    print(f"  Average title relevance: {avg_title_rel:.2f}")
    print(f"  High-scoring windows (≥7): {len(high_score_windows)}")
    print(f"  Title-relevant windows (≥2): {len(title_relevant_windows)}")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    main()
