#!/usr/bin/env python3
"""
Example script demonstrating multi-dataset decomposition.
Run this to test the system with a few problems from each dataset.
"""
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_loaders import create_loader, get_available_datasets
from hcot_decomposer import quick_decompose
from utils import print_statistics, print_execution_plan
import json


def test_dataset_loading():
    """Test loading from different datasets."""
    print("\n" + "="*70)
    print("TESTING DATASET LOADING")
    print("="*70 + "\n")
    
    # List available datasets
    print("Available datasets:")
    for ds in get_available_datasets():
        print(f"  - {ds}")
    print()
    
    # Test Hendrycks MATH dataset
    print("Loading Hendrycks MATH dataset...")
    try:
        loader = create_loader("hendrycks_math")
        
        # Get categories
        categories = loader.get_categories()
        print(f"✓ Found {len(categories)} categories:")
        for cat in categories[:5]:  # Show first 5
            print(f"    - {cat}")
        if len(categories) > 5:
            print(f"    ... and {len(categories) - 5} more")
        
        # Load a few problems
        print("\nLoading 3 sample problems...")
        problems = loader.load(max_problems=3)
        print(f"✓ Loaded {len(problems)} problems\n")
        
        # Display samples
        for i, (problem, answer, metadata) in enumerate(problems, 1):
            print(f"Problem {i}:")
            print(f"  Type: {metadata.get('type', 'Unknown')}")
            print(f"  Level: {metadata.get('level', 'Unknown')}")
            print(f"  Problem: {problem[:100]}...")
            print(f"  Answer: {answer[:50]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nNote: Make sure you have the 'datasets' package installed:")
        print("  pip install datasets --break-system-packages")
        return False


def test_single_decomposition():
    """Test decomposing a single problem."""
    print("\n" + "="*70)
    print("TESTING SINGLE PROBLEM DECOMPOSITION")
    print("="*70 + "\n")
    
    # Use a simple test problem
    test_problem = "Solve the equation 2x + 5 = 13 for x."
    
    print(f"Problem: {test_problem}")
    print("\nChecking Ollama availability...")
    
    try:
        import ollama
        ollama.list()
        print("✓ Ollama is running")
    except:
        print("✗ Ollama not available")
        print("\nPlease start Ollama:")
        print("  ollama serve")
        print("\nThen pull a model:")
        print("  ollama pull llama3.1:latest")
        return False
    
    print("\nDecomposing problem...")
    print("(This may take 10-30 seconds...)\n")
    
    try:
        decomp = quick_decompose(
            problem=test_problem,
            model="llama3.1:latest",
            prompts_path="hcot_prompts.json",
            depth=2,
            branching=2,
            verbose=True
        )
        
        print("\n✓ Decomposition successful!\n")
        print(f"Created {len(decomp.nodes)} sub-problems:")
        for node in decomp.nodes[:5]:  # Show first 5
            print(f"  [{node.id}] {node.goal}")
        
        if len(decomp.nodes) > 5:
            print(f"  ... and {len(decomp.nodes) - 5} more")
        
        # Show statistics
        print_statistics(decomp)
        
        # Show execution plan
        print_execution_plan(decomp)
        
        return True
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nMake sure hcot_prompts.json exists in the current directory.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_category_filtering():
    """Test loading problems from specific categories."""
    print("\n" + "="*70)
    print("TESTING CATEGORY FILTERING")
    print("="*70 + "\n")
    
    try:
        loader = create_loader("hendrycks_math")
        
        # Test filtering by category
        test_categories = ["Algebra", "Geometry"]
        
        for category in test_categories:
            print(f"\nLoading {category} problems...")
            problems = loader.load(max_problems=2, category=category)
            
            if problems:
                print(f"✓ Loaded {len(problems)} {category} problems")
                problem, answer, metadata = problems[0]
                print(f"  Sample: {problem[:80]}...")
            else:
                print(f"  No {category} problems found")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("HCOT MULTI-DATASET SYSTEM - EXAMPLES & TESTS")
    print("="*70)
    
    results = {}
    
    # Test 1: Dataset loading
    results['loading'] = test_dataset_loading()
    
    # Test 2: Category filtering (only if loading worked)
    if results['loading']:
        results['filtering'] = test_category_filtering()
    
    # Test 3: Single decomposition
    results['decomposition'] = test_single_decomposition()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.capitalize():20s} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run batch decomposition:")
        print("     python batch_decompose_universal.py --dataset hendrycks_math --num-problems 10")
        print("  2. Filter by category:")
        print("     python batch_decompose_universal.py --dataset hendrycks_math --category Algebra --num-problems 20")
        print("  3. Use multiple models:")
        print("     python batch_decompose_universal.py --dataset hendrycks_math --models llama3.1:latest qwen2.5:7b")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("  - Ollama not running: ollama serve")
        print("  - Missing datasets package: pip install datasets --break-system-packages")
        print("  - Missing prompts file: Make sure hcot_prompts.json exists")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()