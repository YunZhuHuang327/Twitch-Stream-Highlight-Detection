"""
Merge visual analysis with transcript data by timestamp
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


def parse_timestamp(timestamp_str: str) -> float:
    """
    Convert timestamp string to seconds

    Supports formats:
    - HH:MM:SS
    - MM:SS
    - SS
    """
    parts = timestamp_str.strip().split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(parts[0])


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_visual_analysis(visual_json_path: str) -> List[Dict]:
    """
    Load visual analysis JSON and flatten to list of events

    Returns:
        List of dicts with timestamp_sec, timestamp, type, description, chapter, etc.
    """
    with open(visual_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    events = []

    for chapter_timestamp, analyses in data.items():
        if not analyses:
            continue

        for analysis in analyses:
            if not analysis.get('success', False):
                continue

            timestamp = analysis.get('timestamp', chapter_timestamp)
            timestamp_sec = parse_timestamp(timestamp)

            events.append({
                'timestamp': timestamp,
                'timestamp_sec': timestamp_sec,
                'type': 'VISUAL',
                'chapter': analysis.get('chapter', ''),
                'description': analysis.get('description', ''),
                'model': analysis.get('model', '')
            })

    return events


def load_transcript(transcript_path: str) -> List[Dict]:
    """
    Load merged transcript and parse into list of events

    Supports formats:
    - HH:MM:SS [ASR] text
    - HH:MM:SS [EVENT_LABEL]
    - HH:MM:SS: text (legacy ASR format)

    Returns:
        List of dicts with timestamp_sec, timestamp, type, text, event_type
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    events = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to match timestamp at start of line
        # Pattern: HH:MM:SS or HH:MM:SS:
        match = re.match(r'^(\d{1,2}:\d{2}:\d{2}):?\s*(.*)$', line)
        if not match:
            continue

        timestamp_str = match.group(1)
        rest = match.group(2).strip()

        timestamp_sec = parse_timestamp(timestamp_str)

        # Check if it's labeled format: [ASR] or [EVENT_TYPE]
        label_match = re.match(r'^\[([^\]]+)\]\s*(.*)$', rest)

        if label_match:
            label = label_match.group(1).strip()
            text = label_match.group(2).strip()

            if label == 'ASR':
                event_type = 'ASR'
            else:
                event_type = 'CHAT_EVENT'

            events.append({
                'timestamp': timestamp_str,
                'timestamp_sec': timestamp_sec,
                'type': event_type,
                'text': text,
                'event_label': label if label != 'ASR' else None
            })
        else:
            # Legacy ASR format (no label)
            events.append({
                'timestamp': timestamp_str,
                'timestamp_sec': timestamp_sec,
                'type': 'ASR',
                'text': rest,
                'event_label': None
            })

    return events


def merge_events(visual_events: List[Dict], transcript_events: List[Dict]) -> List[Dict]:
    """
    Merge visual and transcript events, sorted by timestamp

    Returns:
        List of all events sorted by timestamp_sec
    """
    all_events = visual_events + transcript_events
    all_events.sort(key=lambda x: x['timestamp_sec'])
    return all_events


def save_merged_json(events: List[Dict], output_path: str):
    """Save merged events as JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved merged JSON: {output_path}")


def save_merged_text(events: List[Dict], output_path: str):
    """
    Save merged events as human-readable text

    Format:
    HH:MM:SS [TYPE] content
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for event in events:
            timestamp = event['timestamp']
            event_type = event['type']

            if event_type == 'VISUAL':
                chapter = event.get('chapter', '')
                description = event.get('description', '')
                f.write(f"{timestamp} [VISUAL] Chapter: {chapter}\n")
                f.write(f"           {description}\n\n")

            elif event_type == 'ASR':
                text = event.get('text', '')
                f.write(f"{timestamp} [ASR] {text}\n")

            elif event_type == 'CHAT_EVENT':
                label = event.get('event_label', '')
                text = event.get('text', '')
                if text:
                    f.write(f"{timestamp} [{label}] {text}\n")
                else:
                    f.write(f"{timestamp} [{label}]\n")

    print(f"[OK] Saved merged text: {output_path}")


def print_statistics(events: List[Dict]):
    """Print statistics about merged events"""
    total = len(events)
    visual = sum(1 for e in events if e['type'] == 'VISUAL')
    asr = sum(1 for e in events if e['type'] == 'ASR')
    chat = sum(1 for e in events if e['type'] == 'CHAT_EVENT')

    if total > 0:
        start_time = events[0]['timestamp']
        end_time = events[-1]['timestamp']
        duration = events[-1]['timestamp_sec'] - events[0]['timestamp_sec']

        print("\n" + "="*60)
        print("Merge Statistics")
        print("="*60)
        print(f"Total events: {total}")
        print(f"  - Visual analysis: {visual}")
        print(f"  - ASR transcripts: {asr}")
        print(f"  - Chat events: {chat}")
        print(f"\nTime range: {start_time} - {end_time}")
        print(f"Duration: {format_timestamp(duration)}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Merge visual analysis with transcript data by timestamp'
    )
    parser.add_argument('--visual', required=True,
                       help='Path to visual_analysis.json')
    parser.add_argument('--transcript', required=True,
                       help='Path to merged_transcript.txt')
    parser.add_argument('--output', required=True,
                       help='Output directory for merged files')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading visual analysis...")
    visual_events = load_visual_analysis(args.visual)
    print(f"  Loaded {len(visual_events)} visual analysis events")

    print("\nLoading transcript...")
    transcript_events = load_transcript(args.transcript)
    print(f"  Loaded {len(transcript_events)} transcript events")

    print("\nMerging events by timestamp...")
    merged_events = merge_events(visual_events, transcript_events)

    # Print statistics
    print_statistics(merged_events)

    # Save outputs
    print("\nSaving merged data...")
    json_output = output_dir / "merged_events.json"
    text_output = output_dir / "merged_events.txt"

    save_merged_json(merged_events, str(json_output))
    save_merged_text(merged_events, str(text_output))

    print(f"\n[OK] Merge complete!")
    print(f"     JSON: {json_output}")
    print(f"     Text: {text_output}")


if __name__ == "__main__":
    main()
