"""
OpenRouter Client Module

This module provides integration with OpenRouter API for:
1. Text clustering using advanced language models
2. Topic-based summarization
3. Intelligent content analysis
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple
import time

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """Client for interacting with OpenRouter API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "http://localhost:5000"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "Video Analysis Pipeline")
        }
        
        # Default model configurations
        self.clustering_model = os.environ.get("OPENROUTER_CLUSTERING_MODEL", "anthropic/claude-3.5-sonnet")
        self.summary_model = os.environ.get("OPENROUTER_SUMMARY_MODEL", "anthropic/claude-3.5-sonnet")
        
    def _make_request(self, endpoint: str, data: dict, max_retries: int = 3) -> dict:
        """Make a request to OpenRouter API with retry logic"""
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=self.headers, json=data, timeout=120)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    logger.warning(f"Request timeout. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
                    
        raise Exception(f"Failed to get response after {max_retries} attempts")

    def cluster_text_into_topics(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use OpenRouter model to intelligently cluster text segments into coherent topics
        
        Args:
            segments: List of transcript segments with 'text', 'start', 'end' keys
            
        Returns:
            List of topic clusters with metadata
        """
        try:
            # Prepare the text for analysis
            full_text = "\n".join([f"[{seg['start']:.2f}-{seg['end']:.2f}s] {seg['text']}" for seg in segments])
            
            # Create the clustering prompt
            system_prompt = """You are an expert content analyst. Your task is to analyze a video transcript and intelligently group the content into coherent topics.

Instructions:
1. Read through the entire transcript carefully
2. Identify natural topic boundaries based on content shifts
3. Group related segments into coherent topics
4. Each topic should have a clear theme and contain related content
5. Provide meaningful names for each topic
6. Include start/end timestamps for each topic
7. Aim for 3-8 topics depending on content length and complexity

Return your analysis as a JSON object with this exact structure:
{
  "topics": [
    {
      "id": 1,
      "name": "Topic Name",
      "description": "Brief description of what this topic covers",
      "start_time": 0.0,
      "end_time": 120.5,
      "keywords": ["keyword1", "keyword2", "keyword3"],
      "segment_indices": [0, 1, 2, 3]
    }
  ]
}

Make sure the JSON is valid and complete."""

            user_prompt = f"""Analyze this video transcript and cluster it into coherent topics:

{full_text}

Please provide a comprehensive topic analysis following the specified JSON format."""

            data = {
                "model": self.clustering_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 4000
            }
            
            logger.info(f"Sending clustering request to OpenRouter using model: {self.clustering_model}")
            response = self._make_request("chat/completions", data)
            
            # Extract and parse the response
            content = response["choices"][0]["message"]["content"]
            
            # Try to extract JSON from the response
            try:
                # Look for JSON block in the response
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    # Try to find JSON without code blocks
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
                    else:
                        json_text = content
                
                result = json.loads(json_text)
                topics = result.get("topics", [])
                
                # Validate and enrich topics
                for i, topic in enumerate(topics):
                    if "id" not in topic:
                        topic["id"] = i + 1
                    if "segment_indices" not in topic:
                        # Map segments to this topic based on timestamps
                        topic["segment_indices"] = []
                        for j, seg in enumerate(segments):
                            if topic["start_time"] <= seg["start"] <= topic["end_time"]:
                                topic["segment_indices"].append(j)
                    
                    # Ensure required fields exist
                    topic.setdefault("name", f"Topic {topic['id']}")
                    topic.setdefault("description", "Generated topic")
                    topic.setdefault("keywords", [])
                
                logger.info(f"Successfully clustered content into {len(topics)} topics")
                return topics
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {content}")
                raise ValueError("Failed to parse clustering response as JSON")
                
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # Fallback to simple time-based clustering
            return self._fallback_clustering(segments)

    def _fallback_clustering(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback clustering when OpenRouter fails"""
        logger.info("Using fallback time-based clustering")
        
        if not segments:
            return []
        
        # Simple time-based clustering - divide into 3-5 segments
        total_duration = segments[-1]["end"] - segments[0]["start"]
        num_topics = min(5, max(3, len(segments) // 10))
        segment_duration = total_duration / num_topics
        
        topics = []
        for i in range(num_topics):
            start_time = segments[0]["start"] + i * segment_duration
            end_time = start_time + segment_duration
            
            # Find segments in this time range
            segment_indices = []
            for j, seg in enumerate(segments):
                if start_time <= seg["start"] < end_time:
                    segment_indices.append(j)
            
            if segment_indices:
                topics.append({
                    "id": i + 1,
                    "name": f"Topic {i + 1}",
                    "description": f"Content from {start_time:.1f}s to {end_time:.1f}s",
                    "start_time": start_time,
                    "end_time": end_time,
                    "keywords": [],
                    "segment_indices": segment_indices
                })
        
        return topics

    def generate_topic_summary(self, topic: Dict[str, Any], segments: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive summary for a specific topic using OpenRouter
        
        Args:
            topic: Topic metadata with segment indices
            segments: All transcript segments
            
        Returns:
            Detailed summary of the topic
        """
        try:
            # Extract relevant segments for this topic
            topic_segments = [segments[i] for i in topic.get("segment_indices", []) if i < len(segments)]
            
            if not topic_segments:
                return f"Summary for {topic['name']}: No content available."
            
            # Prepare content for summarization
            topic_text = "\n".join([f"[{seg['start']:.2f}s] {seg['text']}" for seg in topic_segments])
            
            system_prompt = """You are an expert content summarizer. Your task is to create a comprehensive, well-structured summary of a specific topic from a video transcript.

Instructions:
1. Create a clear, engaging summary that captures the key points
2. Organize the information logically
3. Include specific details and examples when mentioned
4. Make it informative and accessible
5. Keep it concise but comprehensive (2-4 paragraphs)
6. Focus on the main concepts, explanations, and takeaways

Write the summary in a natural, flowing style that would be helpful for someone who wants to understand this topic without watching the video."""

            user_prompt = f"""Please create a comprehensive summary for the topic "{topic['name']}":

Topic Description: {topic.get('description', 'N/A')}
Keywords: {', '.join(topic.get('keywords', []))}
Duration: {topic.get('start_time', 0):.1f}s to {topic.get('end_time', 0):.1f}s

Content:
{topic_text}

Please provide a detailed summary of this topic."""

            data = {
                "model": self.summary_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            logger.info(f"Generating summary for topic: {topic['name']}")
            response = self._make_request("chat/completions", data)
            
            summary = response["choices"][0]["message"]["content"].strip()
            logger.info(f"Generated summary for topic '{topic['name']}' ({len(summary)} characters)")
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed for topic {topic['name']}: {e}")
            # Fallback summary
            topic_segments = [segments[i] for i in topic.get("segment_indices", []) if i < len(segments)]
            if topic_segments:
                sample_text = " ".join([seg["text"] for seg in topic_segments[:3]])
                return f"Summary for {topic['name']}: {sample_text[:200]}..."
            else:
                return f"Summary for {topic['name']}: Content not available."

    def generate_all_summaries(self, topics: List[Dict[str, Any]], segments: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate summaries for all topics
        
        Args:
            topics: List of topic clusters
            segments: All transcript segments
            
        Returns:
            Dictionary mapping topic IDs to summaries
        """
        summaries = {}
        
        for topic in topics:
            topic_id = str(topic["id"])
            summary = self.generate_topic_summary(topic, segments)
            summaries[topic_id] = summary
            
            # Small delay to respect rate limits
            time.sleep(0.5)
        
        return summaries

def test_openrouter_connection() -> bool:
    """Test if OpenRouter API is accessible"""
    try:
        client = OpenRouterClient()
        # Simple test request
        data = {
            "model": client.clustering_model,
            "messages": [{"role": "user", "content": "Test connection. Respond with 'OK'."}],
            "max_tokens": 10
        }
        response = client._make_request("chat/completions", data)
        return "OK" in response["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenRouter connection test failed: {e}")
        return False