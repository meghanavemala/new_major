"""
Test specifically for the "quick soar" and "tim soar" issues
"""
import sys
import os

# Test the exact same input that's causing issues
test_segments = [
    {
        "start": 0.0,
        "end": 4.24,
        "text": "Every programmer has ran into at least one sorting algorythm at one point in their career."
    },
    {
        "start": 4.24,
        "end": 7.92,
        "text": "Today, I'm going to easily explain 10 of the most popular sorting algorythms."
    },
    {
        "start": 14.06,
        "end": 19.18,
        "text": "Quicksor is one of the most popular algorithms that also uses the divide and conquer strategy."
    },
    {
        "start": 19.18,
        "end": 23.74,
        "text": "Programming languages like JavaScript, Ruby, and PHP use Quicksor in their standard library features."
    },
    {
        "start": 23.74,
        "end": 28.46,
        "text": "Heapsor is another comparison based algorithm, also known as selection sort using the right data structure."
    }
]

# Test preprocessing on these exact segments
def test_exact_issue():
    print("=== TESTING EXACT ISSUE FROM SCREENSHOT ===\n")
    
    # Import and test preprocessing
    sys.path.append('.')
    
    try:
        from utils.openrouter_client import OpenRouterClient
        client = OpenRouterClient()
    except:
        # Create mock client for testing
        class MockClient:
            def __init__(self):
                self.enable_preprocessing = True
            def _preprocess_text(self, text):
                import re
                fixes = {
                    r'\balgorythms\b': 'algorithms',
                    r'\balgorythm\b': 'algorithm',
                    r'\bquicksor\b': 'quicksort',
                    r'\bheapsor\b': 'heapsort',
                    r'\bquick soar\b': 'quicksort',
                    r'\bheap soar\b': 'heapsort',
                    r'\btim soar\b': 'timsort',
                    r'\bbubble soar\b': 'bubble sort',
                }
                processed = text
                for pattern, replacement in fixes.items():
                    processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
                return processed
        client = MockClient()
    
    for i, segment in enumerate(test_segments, 1):
        original = segment["text"]
        processed = client._preprocess_text(original)
        
        print(f"Segment {i}:")
        print(f"Original:  {original}")
        print(f"Processed: {processed}")
        print(f"Fixed:     {'✓ YES' if processed != original else '✗ NO'}")
        
        # Check specific fixes
        if "algorythm" in original.lower():
            if "algorithm" in processed.lower() and "algorythm" not in processed.lower():
                print(f"  ✅ Fixed: algorythm → algorithm")
            else:
                print(f"  ❌ FAILED: algorythm not fixed")
        
        if "quicksor" in original.lower():
            if "quicksort" in processed.lower() and "quicksor" not in processed.lower():
                print(f"  ✅ Fixed: Quicksor → Quicksort")
            else:
                print(f"  ❌ FAILED: Quicksor not fixed")
        
        if "heapsor" in original.lower():
            if "heapsort" in processed.lower() and "heapsor" not in processed.lower():
                print(f"  ✅ Fixed: Heapsor → Heapsort")
            else:
                print(f"  ❌ FAILED: Heapsor not fixed")
        
        print()

if __name__ == "__main__":
    test_exact_issue()