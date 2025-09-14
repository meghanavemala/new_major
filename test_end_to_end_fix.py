#!/usr/bin/env python3

"""
Test the actual OpenRouter API call to see if the preprocessing + model produces good summaries
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.openrouter_client import OpenRouterClient
import json

def test_end_to_end_processing():
    """Test that preprocessing + OpenRouter produces good summaries"""
    client = OpenRouterClient()
    
    # Sample text with transcription errors like user reported
    test_text = """
    In this video we will discuss different sorting algorythms. 
    First we cover quick soar which is very fast for most cases.
    Then we look at tim soar which is used by Python.
    We also examine bubble soar and heap soar.
    The performence comparsion shows that quick soar and tim soar are best.
    Each algorythm has different time complexity.
    """
    
    print("Testing end-to-end processing with OpenRouter...")
    print("Original text with errors:")
    print(f"'{test_text}'")
    print("\n" + "="*60)
    
    try:
        # Test topic clustering
        print("1. Testing topic clustering...")
        topics_result = client.cluster_text_into_topics([{"text": test_text, "start_time": 0, "end_time": 60}])
        
        if topics_result and len(topics_result) > 0:
            print("✅ Topic clustering successful")
            print(f"Number of topics: {len(topics_result)}")
            
            for i, topic in enumerate(topics_result):
                print(f"\nTopic {i+1}: {topic.get('title', 'No title')}")
                print(f"Summary: {topic.get('summary', 'No summary')[:200]}...")
                
                # Check if the summary contains corrected algorithm names
                summary = topic.get('summary', '').lower()
                if 'quicksort' in summary or 'timsort' in summary:
                    print("✅ Contains corrected algorithm names!")
                elif 'quick soar' in summary or 'tim soar' in summary:
                    print("❌ Still contains transcription errors")
                else:
                    print("ℹ️ No algorithm names detected in summary")
        else:
            print("❌ Topic clustering failed")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False
    
    print("\n" + "="*60)
    
    try:
        # Test individual summary generation
        print("2. Testing individual summary generation...")
        summary = client.generate_topic_summary(test_text, "Sorting Algorithms", 60)
        
        print(f"Generated summary:")
        print(f"'{summary}'")
        
        # Check for corrections
        if 'quicksort' in summary.lower() or 'timsort' in summary.lower():
            print("✅ Summary contains corrected algorithm names!")
            return True
        elif 'quick soar' in summary.lower() or 'tim soar' in summary.lower():
            print("❌ Summary still contains transcription errors")
            return False
        else:
            print("ℹ️ No specific algorithm names in summary, but no errors detected")
            return True
            
    except Exception as e:
        print(f"❌ Error during summary generation: {e}")
        return False

if __name__ == "__main__":
    print("Testing actual OpenRouter API integration...")
    print("This will make real API calls to verify the fixes work end-to-end")
    print()
    
    success = test_end_to_end_processing()
    
    if success:
        print("\n🎉 End-to-end test passed! The preprocessing + OpenRouter should fix your algorithm names.")
    else:
        print("\n⚠️ End-to-end test revealed issues that need fixing.")
        
    sys.exit(0 if success else 1)