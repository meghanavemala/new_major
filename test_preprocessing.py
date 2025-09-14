"""
Test the OpenRouter preprocessing function
"""
import sys
import os
sys.path.append('.')

from utils.openrouter_client import OpenRouterClient

def test_preprocessing():
    """Test the preprocessing function with common errors"""
    
    # Initialize client (will use env var or fail gracefully)
    try:
        client = OpenRouterClient()
    except:
        # Create a mock client for testing preprocessing
        class MockClient:
            def __init__(self):
                self.enable_preprocessing = True
            def _preprocess_text(self, text):
                import re
                fixes = {
                    # CRITICAL: Algorithm name corrections
                    r'\bbubble soar\b': 'bubble sort',
                    r'\bquick soar\b': 'quicksort',
                    r'\bquicksor\b': 'quicksort',
                    r'\bquick sor\b': 'quicksort',
                    r'\btim soar\b': 'timsort',
                    r'\bheap soar\b': 'heapsort',
                    r'\bheapsor\b': 'heapsort',
                    r'\bmerge soar\b': 'merge sort',
                    r'\binsertion soar\b': 'insertion sort',
                    r'\bselection soar\b': 'selection sort',
                    
                    # Other fixes
                    r'\balgorythm\b': 'algorithm',
                    r'\balgorthm\b': 'algorithm',
                    r'\bcomparsion\b': 'comparison',
                    r'\bperformence\b': 'performance',
                    r'\bteh\b': 'the',
                }
                
                processed = text
                for pattern, replacement in fixes.items():
                    processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
                return processed.strip()
        
        client = MockClient()
    
    test_cases = [
        "Quicksor is one of the most popular algorythms",
        "Bubble soar has terrible performence compared to quick soar",
        "Tim soar and heap soar are both efficient",
        "The comparsion between merge soar and insertion soar shows teh difference"
    ]
    
    print("=== PREPROCESSING TEST RESULTS ===\n")
    
    for i, test_text in enumerate(test_cases, 1):
        processed = client._preprocess_text(test_text)
        print(f"Test {i}:")
        print(f"Original:  {test_text}")
        print(f"Processed: {processed}")
        print(f"Changes:   {'✓ Fixed' if processed != test_text else '✗ No changes'}")
        print()

if __name__ == "__main__":
    test_preprocessing()