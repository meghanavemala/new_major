#!/usr/bin/env python3
"""
Test script to demonstrate the new user prompt feature for topic clustering and analysis.
This shows how the user prompt guides the AI to focus on specific aspects of the content.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.topic_analyzer import analyze_topic_segments
from utils.summarizer import summarize_cluster

def test_prompt_feature():
    """Test the user prompt feature with sample data."""
    
    # Sample transcript segments (simulating a technical tutorial video)
    sample_segments = [
        {
            'text': 'In this tutorial, we will learn how to implement machine learning algorithms using Python.',
            'start': 0.0,
            'end': 5.0
        },
        {
            'text': 'First, let\'s install the required libraries: scikit-learn, pandas, and numpy.',
            'start': 5.0,
            'end': 10.0
        },
        {
            'text': 'The scikit-learn library provides excellent tools for data preprocessing and model training.',
            'start': 10.0,
            'end': 15.0
        },
        {
            'text': 'Now, let\'s load our dataset and perform some exploratory data analysis.',
            'start': 15.0,
            'end': 20.0
        },
        {
            'text': 'We can see that our data has some missing values that need to be handled.',
            'start': 20.0,
            'end': 25.0
        },
        {
            'text': 'The practical applications of this algorithm include fraud detection and recommendation systems.',
            'start': 25.0,
            'end': 30.0
        },
        {
            'text': 'Let\'s implement the random forest algorithm step by step.',
            'start': 30.0,
            'end': 35.0
        },
        {
            'text': 'The key insights from our analysis show that feature engineering is crucial for model performance.',
            'start': 35.0,
            'end': 40.0
        }
    ]
    
    print("🧪 Testing User Prompt Feature")
    print("=" * 50)
    
    # Test 1: No prompt (default behavior)
    print("\n📋 Test 1: No User Prompt (Default Analysis)")
    print("-" * 40)
    topics_no_prompt = analyze_topic_segments(sample_segments, language='en')
    for i, topic in enumerate(topics_no_prompt):
        print(f"Topic {i+1}: {topic['name']}")
        print(f"  Keywords: {', '.join(topic['keywords'][:5])}")
        print(f"  Duration: {topic['duration']:.1f}s")
        print()
    
    # Test 2: Technical focus prompt
    print("\n🔧 Test 2: Technical Focus Prompt")
    print("-" * 40)
    technical_prompt = "Focus on technical concepts, implementation steps, and code examples"
    topics_technical = analyze_topic_segments(sample_segments, language='en', user_prompt=technical_prompt)
    for i, topic in enumerate(topics_technical):
        print(f"Topic {i+1}: {topic['name']}")
        print(f"  Keywords: {', '.join(topic['keywords'][:5])}")
        print(f"  Duration: {topic['duration']:.1f}s")
        print()
    
    # Test 3: Practical applications focus
    print("\n💡 Test 3: Practical Applications Focus")
    print("-" * 40)
    practical_prompt = "Highlight practical applications, real-world examples, and business use cases"
    topics_practical = analyze_topic_segments(sample_segments, language='en', user_prompt=practical_prompt)
    for i, topic in enumerate(topics_practical):
        print(f"Topic {i+1}: {topic['name']}")
        print(f"  Keywords: {', '.join(topic['keywords'][:5])}")
        print(f"  Duration: {topic['duration']:.1f}s")
        print()
    
    # Test 4: Summarization with prompts
    print("\n📝 Test 4: Summarization with User Prompts")
    print("-" * 40)
    
    # Create a sample cluster
    sample_cluster = sample_segments[:4]  # First 4 segments
    
    print("Default Summary:")
    default_summary = summarize_cluster(sample_cluster, language='en')
    print(f"  {default_summary}")
    print()
    
    print("Technical Focus Summary:")
    tech_summary = summarize_cluster(sample_cluster, language='en', user_prompt=technical_prompt)
    print(f"  {tech_summary}")
    print()
    
    print("Practical Focus Summary:")
    practical_summary = summarize_cluster(sample_cluster, language='en', user_prompt=practical_prompt)
    print(f"  {practical_summary}")
    print()
    
    print("✅ User Prompt Feature Test Completed!")
    print("\n🎯 Key Benefits:")
    print("• Custom topic clustering based on user interests")
    print("• Focused summarization on specific aspects")
    print("• Better content organization for different use cases")
    print("• Enhanced topic naming with context awareness")

if __name__ == "__main__":
    test_prompt_feature()
