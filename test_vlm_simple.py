"""
Simple test script for VLM Analyzer

Tests the VLM analyzer with your existing chapters file.
"""

import os
import json
from pathlib import Path
from tools.vlm_analyzer import VLMAnalyzer


def main():
    # Configuration
    video_path = "123.mp4"
    chapters_file = "outputs/chapters/123/chapters.json"
    output_dir = "outputs/visual_analysis/123"
    api_key = os.getenv('OPENAI_API_KEY')

    # Check files exist
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return

    if not Path(chapters_file).exists():
        print(f"❌ Chapters file not found: {chapters_file}")
        print("Please run chapter segmentation first:")
        print("  python quick_chapter.py --api openai --yes")
        return

    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Set it with: set OPENAI_API_KEY=your-key-here")
        return

    print("="*60)
    print("🎨 VLM Analyzer Test")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Chapters: {chapters_file}")
    print(f"Output: {output_dir}")
    print(f"Model: gpt-4o-mini")
    print("="*60)

    # Load chapters
    with open(chapters_file, 'r', encoding='utf-8') as f:
        chapters = json.load(f)

    print(f"\n✓ Loaded {len(chapters)} chapters")

    # Initialize analyzer
    analyzer = VLMAnalyzer(model="gpt-4o-mini", api_key=api_key)

    # Analyze (test with first 3 chapters only, 3 frames each)
    test_chapters = dict(list(chapters.items()))#[:3]

    print("\n⚠️  Testing with first 3 chapters only (3 frames each)")
    print("This is a quick test. Full analysis will process all chapters.\n")

    results = analyzer.analyze_video_chapters(
        video_path=video_path,
        chapters=test_chapters,
        output_dir=output_dir,
        frames_per_chapter=3,
        video_context="TwitchCon livestream with Agent00, exploring Little Italy in San Diego"
    )

    # Display sample results
    print("\n" + "="*60)
    print("Sample Results:")
    print("="*60)

    for timestamp, chapter_results in list(results.items())[:2]:
        print(f"\n📍 {timestamp}: {chapters[timestamp]}")
        for i, result in enumerate(chapter_results[:2]):  # Show first 2 frames
            if result['success']:
                print(f"\n  Frame {i+1} ({result['timestamp']}):")
                print(f"  {result['description'][:200]}...")

    print("\n" + "="*60)
    print("✓ Test complete!")
    print(f"Full results saved to: {output_dir}/visual_analysis.json")
    print("="*60)

    # Estimate cost for full analysis
    total_frames = len(chapters) * 3
    cost_per_frame = 0.0015  # USD for gpt-4o-mini
    total_cost_usd = total_frames * cost_per_frame
    total_cost_twd = total_cost_usd * 32

    print(f"\n💰 Cost estimate for full analysis:")
    print(f"   Chapters: {len(chapters)}")
    print(f"   Frames: {total_frames} (3 per chapter)")
    print(f"   Cost: ${total_cost_usd:.2f} USD (~NT$ {total_cost_twd:.1f})")

    print("\n🚀 To run full analysis:")
    print("   Modify this script to use all chapters (remove [:3] limit)")


if __name__ == "__main__":
    main()
