#!/usr/bin/env python3

"""
Comprehensive test of preprocessing fixes for algorithm names and other transcription errors
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.openrouter_client import OpenRouterClient

def test_preprocessing():
    """Test all preprocessing fixes"""
    client = OpenRouterClient()
    
    test_cases = [
        # Algorithm names with various cases
        ("The quick soar algorithm is fast", "The quicksort algorithm is fast"),
        ("Quick Soar is faster than bubble sort", "Quicksort is faster than bubble sort"),
        ("Quicksor implementation", "Quicksort implementation"),
        ("tim soar beats quick soar", "timsort beats quicksort"),
        ("Tim Soar algorithm", "Timsort algorithm"),
        ("Heap Soar vs bubble soar", "Heapsort vs bubble sort"),
        ("merge soar divide and conquer", "merge sort divide and conquer"),
        
        # Common transcription errors
        ("algorythm implementation", "algorithm implementation"),
        ("the algorthm is efficient", "the algorithm is efficient"),
        ("comparsion of algorithms", "comparison of algorithms"),
        ("performence metrics", "performance metrics"),
        
        # Mixed cases
        ("Quick Soar and Tim Soar are the fastest algorythms", "Quicksort and Timsort are the fastest algorithms"),
        ("bubble soar vs quick soar vs tim soar", "bubble sort vs quicksort vs timsort"),
        
        # Edge cases
        ("quicksort", "quicksort"),  # Should remain unchanged
        ("Quicksort", "Quicksort"),  # Should remain unchanged
        ("Quick Sort", "Quick Sort"),  # Should remain unchanged
    ]
    
    print("Testing comprehensive preprocessing...")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = client._preprocess_text(input_text)
        
        print(f"\nTest {i}:")
        print(f"Input:    '{input_text}'")
        print(f"Expected: '{expected}'")
        print(f"Got:      '{result}'")
        
        if result == expected:
            print("✅ PASS")
            passed += 1
        else:
            print("❌ FAIL")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All preprocessing tests passed!")
        return True
    else:
        print(f"⚠️  {failed} tests failed")
        return False

if __name__ == "__main__":
    success = test_preprocessing()
    sys.exit(0 if success else 1)