#!/usr/bin/env python3
"""Score existing decompositions in the dataset."""
from __future__ import annotations

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pathlib import Path
import json
from decomposition_scorer import evaluate_dataset, score_decomposition
from utils import load_decomposition


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Score existing decompositions")
    parser.add_argument("--dataset-dir", default="data/decompositions", help="Dataset directory")
    parser.add_argument("--output", default=None, help="Output file for scores (default: print to stdout)")
    parser.add_argument("--threshold", type=float, default=0.65, help="Quality threshold for filtering")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("SCORING EXISTING DECOMPOSITIONS")
    print(f"{'='*70}\n")
    
    dataset_path = Path(args.dataset_dir)
    
    if not dataset_path.exists():
        print(f"✗ Dataset not found: {dataset_path}")
        return 1
    
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        summary = evaluate_dataset(str(dataset_path))
    except Exception as e:
        print(f"✗ Error evaluating dataset: {e}")
        return 1
    
    # Print summary
    print(f"\n{'='*70}")
    print("SCORING RESULTS")
    print(f"{'='*70}")
    print(f"Total decompositions:     {summary['total_decompositions']}")
    print(f"Average Q_total:          {summary['avg_Q_total']:.3f}")
    print(f"High quality (≥0.65):     {summary['high_quality_count']} ({summary['high_quality_count']/summary['total_decompositions']*100:.1f}%)")
    print(f"Low quality (<0.50):      {summary['low_quality_count']} ({summary['low_quality_count']/summary['total_decompositions']*100:.1f}%)")
    print(f"{'='*70}\n")
    
    # Show top and bottom decompositions
    scores_sorted = sorted(summary['individual_scores'], key=lambda x: x['Q_total'], reverse=True)
    
    print("Top 5 Best Quality:")
    for i, s in enumerate(scores_sorted[:5], 1):
        print(f"  {i}. {s['decomp_id']}: Q={s['Q_total']:.3f} (coverage={s['coverage']:.3f}, check={s['check_rate']:.3f})")
    
    print("\nBottom 5 Worst Quality:")
    for i, s in enumerate(scores_sorted[-5:], 1):
        print(f"  {i}. {s['decomp_id']}: Q={s['Q_total']:.3f} (coverage={s['coverage']:.3f}, check={s['check_rate']:.3f})")
    
    # Filter by threshold
    above_threshold = [s for s in summary['individual_scores'] if s['Q_total'] >= args.threshold]
    below_threshold = [s for s in summary['individual_scores'] if s['Q_total'] < args.threshold]
    
    print(f"\n{'='*70}")
    print(f"FILTERING BY THRESHOLD: {args.threshold}")
    print(f"{'='*70}")
    print(f"Above threshold: {len(above_threshold)}")
    print(f"Below threshold: {len(below_threshold)}")
    
    if below_threshold:
        print("\nDecompositions below threshold:")
        for s in below_threshold:
            print(f"  - {s['decomp_id']}: Q={s['Q_total']:.3f}")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    print(f"\n{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())