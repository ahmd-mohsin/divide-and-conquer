#!/usr/bin/env python3
"""
Example script demonstrating all supported MATH datasets.
Tests: Hendrycks MATH, MiniF2F, ProofNet, and DeepMind.
"""
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_loaders import create_loader, get_available_datasets


def test_hendrycks_math():
    """Test Hendrycks MATH competition dataset."""
    print("\n" + "="*70)
    print("TESTING: Hendrycks MATH Dataset")
    print("="*70 + "\n")
    
    try:
        loader = create_loader("hendrycks_math")
        
        # Get categories
        categories = loader.get_categories()
        print(f"✓ Found {len(categories)} categories:")
        for cat in categories[:3]:
            print(f"    - {cat}")
        if len(categories) > 3:
            print(f"    ... and {len(categories) - 3} more")
        
        # Load sample problems
        print("\nLoading 2 sample problems...")
        problems = loader.load(max_problems=2)
        print(f"✓ Loaded {len(problems)} problems\n")
        
        for i, (problem, answer, metadata) in enumerate(problems, 1):
            print(f"Problem {i}:")
            print(f"  Type: {metadata.get('type', 'Unknown')}")
            print(f"  Level: {metadata.get('level', 'Unknown')}")
            print(f"  Problem: {problem[:80]}...")
            print(f"  Answer: {answer[:50]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_minif2f():
    """Test MiniF2F formal mathematics dataset."""
    print("\n" + "="*70)
    print("TESTING: MiniF2F Dataset")
    print("="*70 + "\n")
    
    try:
        loader = create_loader("minif2f")
        
        # Get categories (splits)
        categories = loader.get_categories()
        print(f"✓ Found {len(categories)} splits:")
        for cat in categories:
            print(f"    - {cat}")
        
        # Load sample problems
        print("\nLoading 2 sample problems...")
        problems = loader.load(max_problems=2)
        print(f"✓ Loaded {len(problems)} problems\n")
        
        for i, (problem, answer, metadata) in enumerate(problems, 1):
            print(f"Problem {i}:")
            print(f"  Name: {metadata.get('name', 'Unknown')}")
            print(f"  Split: {metadata.get('level', 'Unknown')}")
            print(f"  Problem: {problem[:100]}...")
            print(f"  Goal: {answer[:80]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nNote: Make sure you have access to the dataset.")
        print("You may need to login: huggingface-cli login")
        return False


def test_proofnet():
    """Test ProofNet undergraduate theorem proving dataset."""
    print("\n" + "="*70)
    print("TESTING: ProofNet Dataset")
    print("="*70 + "\n")
    
    try:
        loader = create_loader("proofnet")
        
        # Get categories
        categories = loader.get_categories()
        print(f"✓ Found {len(categories)} problem sources:")
        for cat in categories[:5]:
            print(f"    - {cat}")
        if len(categories) > 5:
            print(f"    ... and {len(categories) - 5} more")
        
        # Load sample problems
        print("\nLoading 2 sample problems...")
        problems = loader.load(max_problems=2)
        print(f"✓ Loaded {len(problems)} problems\n")
        
        for i, (problem, answer, metadata) in enumerate(problems, 1):
            print(f"Problem {i}:")
            print(f"  Source: {metadata.get('type', 'Unknown')}")
            print(f"  ID: {metadata.get('problem_id', 'Unknown')}")
            print(f"  Problem: {problem[:80]}...")
            print(f"  Proof available: {'Yes' if answer != '[Proof not available]' else 'No'}")
            print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_deepmind():
    """Test DeepMind legacy dataset."""
    print("\n" + "="*70)
    print("TESTING: DeepMind Dataset (Legacy)")
    print("="*70 + "\n")
    
    try:
        loader = create_loader("deepmind")
        
        # Load sample problems
        print("Loading 2 sample problems...")
        problems = loader.load(max_problems=2)
        print(f"✓ Loaded {len(problems)} problems\n")
        
        for i, (problem, answer, metadata) in enumerate(problems, 1):
            print(f"Problem {i}:")
            print(f"  Problem: {problem}")
            print(f"  Answer: {answer}")
            print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("HCOT MULTI-DATASET SYSTEM - COMPREHENSIVE TEST")
    print("="*70)
    
    print("\nAvailable datasets:")
    for ds in get_available_datasets():
        print(f"  - {ds}")
    
    results = {}
    
    # Test all datasets
    results['hendrycks_math'] = test_hendrycks_math()
    results['minif2f'] = test_minif2f()
    results['proofnet'] = test_proofnet()
    results['deepmind'] = test_deepmind()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for dataset, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{dataset:20s} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All datasets are working!")
    else:
        print("\n⚠ Some datasets failed. See errors above.")
    
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    print("""
# Hendrycks MATH (Competition Problems)
python batch_decompose_universal.py --dataset hendrycks_math --category Algebra --num-problems 10

# MiniF2F (Formal Mathematics)
python batch_decompose_universal.py --dataset minif2f --num-problems 10

# ProofNet (Undergraduate Theorems)
python batch_decompose_universal.py --dataset proofnet --num-problems 10

# Filter ProofNet by source (e.g., Rudin problems)
python batch_decompose_universal.py --dataset proofnet --category Rudin --num-problems 5

# Compare all datasets
python batch_decompose_universal.py --dataset hendrycks_math --num-problems 3
python batch_decompose_universal.py --dataset minif2f --num-problems 3
python batch_decompose_universal.py --dataset proofnet --num-problems 3
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()