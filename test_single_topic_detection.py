import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from utils.topic_analyzer import should_force_single_topic

def test_single_topic_detection():
    """Test the single-topic detection function with various prompts."""
    
    # Test cases that should return True (single topic)
    single_topic_prompts = [
        "Explain quantum physics",
        "What is machine learning?",
        "Focus only on climate change",
        "Just tell me about renewable energy",
        "Limit to artificial intelligence applications",
        "Stick to the main points about blockchain",
        "How to cook pasta",
        "Why is the sky blue?",
        "Tell me about the history of Rome"
    ]
    
    # Test cases that should return False (multiple topics)
    multi_topic_prompts = [
        "Discuss both climate change and renewable energy",
        "Compare artificial intelligence and machine learning",
        "Talk about history, politics, and economics",
        "Cover all aspects of the topic",
        "Explain various concepts in computer science",
        "Describe different approaches to problem solving"
    ]
    
    print("Testing single-topic prompts:")
    for prompt in single_topic_prompts:
        result = should_force_single_topic(prompt)
        print(f"  '{prompt}' -> {result}")
        if not result:
            print(f"    ERROR: Expected True but got False")
    
    print("\nTesting multi-topic prompts:")
    for prompt in multi_topic_prompts:
        result = should_force_single_topic(prompt)
        print(f"  '{prompt}' -> {result}")
        if result:
            print(f"    ERROR: Expected False but got True")
    
    print("\nTesting edge cases:")
    edge_cases = [
        "",  # Empty string
        None,  # None value
        "   ",  # Whitespace only
        "Random text without specific focus"  # No clear focus indicators
    ]
    
    for prompt in edge_cases:
        result = should_force_single_topic(prompt)
        print(f"  '{prompt}' -> {result}")

if __name__ == "__main__":
    test_single_topic_detection()