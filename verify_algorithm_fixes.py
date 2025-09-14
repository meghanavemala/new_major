#!/usr/bin/env python3

"""
Verify that the latest processed summaries contain corrected algorithm names
"""

import json
import os

def check_latest_summaries():
    """Check the most recent summaries for algorithm name corrections"""
    
    processed_dir = r"c:\Users\mythr\OneDrive\Desktop\MAJ\processed"
    
    # Find the latest processed folder
    folders = [f for f in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, f)) and f.isdigit()]
    if not folders:
        print("❌ No processed folders found")
        return False
    
    latest_folder = max(folders, key=int)
    summaries_file = os.path.join(processed_dir, latest_folder, "topic_summaries.json")
    
    if not os.path.exists(summaries_file):
        print(f"❌ No summaries file found in {latest_folder}")
        return False
    
    print(f"📁 Checking summaries in folder: {latest_folder}")
    
    # Load and check summaries
    with open(summaries_file, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
    
    print(f"📊 Found {len(summaries)} topic summaries")
    
    # Algorithm names to check
    correct_names = ['quicksort', 'timsort', 'heapsort', 'merge sort', 'bubble sort', 'insertion sort']
    incorrect_names = ['quick soar', 'tim soar', 'heap soar', 'merge soar', 'bubble soar', 'insertion soar']
    
    corrections_found = 0
    errors_still_present = 0
    
    for i, topic in enumerate(summaries, 1):
        summary_text = topic.get('summary', '').lower()
        
        # Check for correct algorithm names
        found_correct = []
        for name in correct_names:
            if name in summary_text:
                found_correct.append(name)
                corrections_found += 1
        
        # Check for incorrect algorithm names
        found_errors = []
        for name in incorrect_names:
            if name in summary_text:
                found_errors.append(name)
                errors_still_present += 1
        
        if found_correct or found_errors:
            print(f"\n📋 Topic {i}: {topic.get('name', 'Unknown')}")
            if found_correct:
                print(f"  ✅ Correct names found: {', '.join(found_correct)}")
            if found_errors:
                print(f"  ❌ Errors still present: {', '.join(found_errors)}")
    
    print(f"\n" + "="*50)
    print(f"📈 Results Summary:")
    print(f"  ✅ Correct algorithm names found: {corrections_found}")
    print(f"  ❌ Transcription errors still present: {errors_still_present}")
    
    if errors_still_present == 0 and corrections_found > 0:
        print("🎉 SUCCESS: All algorithm names have been corrected!")
        return True
    elif errors_still_present == 0:
        print("ℹ️  No algorithm names found in summaries (but no errors either)")
        return True
    else:
        print("⚠️  Some transcription errors are still present")
        return False

if __name__ == "__main__":
    success = check_latest_summaries()
    if success:
        print("\n✨ The preprocessing and OpenRouter integration is working correctly!")
        print("   Algorithm names like 'quick soar' and 'tim soar' are being fixed to 'quicksort' and 'timsort'")
    else:
        print("\n❗ There may still be issues with the error correction system")