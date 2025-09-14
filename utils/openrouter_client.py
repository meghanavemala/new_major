"""
OpenRouter Client Module

This module provides integration with OpenRouter API for:
1. Text clustering using adva                # Sorting algorithm names - CRITICAL FIXES (with proper capitalization)
                r'\bBubble [Ss]oar\b': 'Bubble Sort',
                r'\bbubble soar\b': 'bubble sort',
                r'\bQuick [Ss]oar\b': 'Quicksort', 
                r'\bquick soar\b': 'quicksort',
                r'\bQuicksor\b': 'Quicksort',
                r'\bquicksor\b': 'quicksort',
                r'\bQuick [Ss]or\b': 'Quicksort',
                r'\bquick sor\b': 'quicksort',
                r'\bTim [Ss]oar\b': 'Timsort',
                r'\btim soar\b': 'timsort',
                r'\bTim [Ss]ort\b': 'Timsort',
                r'\btim sort\b': 'timsort',
                r'\bHeap [Ss]oar\b': 'Heapsort',
                r'\bheap soar\b': 'heapsort',
                r'\bHeapsor\b': 'Heapsort',
                r'\bheapsor\b': 'heapsort',
                r'\bMerge [Ss]oar\b': 'Merge Sort',
                r'\bmerge soar\b': 'merge sort',
                r'\bInsertion [Ss]oar\b': 'Insertion Sort',
                r'\binsertion soar\b': 'insertion sort',
                r'\bSelection [Ss]oar\b': 'Selection Sort',
                r'\bselection soar\b': 'selection sort',e models
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
        
        # Error correction settings
        self.enable_preprocessing = os.environ.get("OPENROUTER_PREPROCESS_TEXT", "true").lower() == "true"
        
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

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text to fix obvious transcription errors"""
        if not self.enable_preprocessing:
            logger.info("Text preprocessing disabled")
            return text
        
        logger.debug(f"Preprocessing text: {text[:100]}...")
            
        try:
            import re
            
            # Common transcription error fixes
            fixes = {
                # Algorithm-related terms
                r'\balgorythms\b': 'algorithms',
                r'\balgorythm\b': 'algorithm',
                r'\balgorthms\b': 'algorithms',
                r'\balgorthm\b': 'algorithm',
                r'\balgorith\b': 'algorithm',
                r'\balgo\b': 'algorithm',
                
                # Performance terms
                r'\bcomparsion\b': 'comparison',
                r'\bperformence\b': 'performance',
                r'\befficienty\b': 'efficiency',
                r'\boptimisation\b': 'optimization',
                
                # Implementation terms
                r'\bimplemention\b': 'implementation',
                r'\bexampel\b': 'example',
                r'\bfuntion\b': 'function',
                r'\bmetod\b': 'method',
                
                # Common words
                r'\bteh\b': 'the',
                r'\band and\b': 'and',
                r'\btha\b': 'the',
                r'\bwith with\b': 'with',
                r'\bis is\b': 'is',
                r'\bwill will\b': 'will',
                r'\bcan can\b': 'can',
                r'\bin in\b': 'in',
                r'\bto to\b': 'to',
                
                # Sorting algorithm names - CRITICAL FIXES (with proper capitalization)
                r'\bBubble [Ss]oar\b': 'Bubble Sort',
                r'\bbubble soar\b': 'bubble sort',
                r'\bQuick [Ss]oar\b': 'Quicksort', 
                r'\bquick soar\b': 'quicksort',
                r'\bQuicksor\b': 'Quicksort',
                r'\bquicksor\b': 'quicksort',
                r'\bQuick [Ss]or\b': 'Quicksort',
                r'\bquick sor\b': 'quicksort',
                r'\bTim [Ss]oar\b': 'Timsort',
                r'\btim soar\b': 'timsort',
                r'\bTim [Ss]ort\b': 'Timsort', 
                r'\btim sort\b': 'timsort',
                r'\bHeap [Ss]oar\b': 'Heapsort',
                r'\bheap soar\b': 'heapsort',
                r'\bHeapsor\b': 'Heapsort',
                r'\bheapsor\b': 'heapsort',
                r'\bMerge [Ss]oar\b': 'Merge Sort',
                r'\bmerge soar\b': 'merge sort',
                r'\bInsertion [Ss]oar\b': 'Insertion Sort',
                r'\binsertion soar\b': 'insertion sort',
                r'\bSelection [Ss]oar\b': 'Selection Sort',
                r'\bselection soar\b': 'selection sort',
                
                # General sorting terms
                r'\bsort sort\b': 'sort',
                r'\bbuble\b': 'bubble',
                r'\bquick sort\b': 'quicksort',
                r'\bmerge sort\b': 'merge sort',
                r'\bheap sort\b': 'heapsort',
                
                # Programming terms
                r'\baray\b': 'array',
                r'\belist\b': 'list',
                r'\bvariabel\b': 'variable',
                r'\bindx\b': 'index',
                
                # Remove filler words and fix spacing
                r'\s+um\s+': ' ',
                r'\s+uh\s+': ' ',
                r'\s+like\s+': ' ',
                r'\s+you know\s+': ' ',
                
                # Fix repeated words (general pattern)
                r'\b(\w+)\s+\1\b': r'\1',
                
                # Fix spacing around punctuation
                r'\s+([,.!?;:])': r'\1',
                r'([,.!?;:])\s*([A-Z])': r'\1 \2',
                
                # Fix multiple spaces
                r'\s+': ' '
            }
            
            processed = text
            changes_made = []
            for pattern, replacement in fixes.items():
                old_processed = processed
                processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
                if processed != old_processed:
                    changes_made.append(f"{pattern} -> {replacement}")
            
            if changes_made:
                logger.info(f"Text preprocessing made {len(changes_made)} corrections")
            
            return processed.strip()
            
        except Exception as e:
            logger.warning(f"Text preprocessing failed: {e}")
            return text

    def cluster_text_into_topics(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use OpenRouter model to intelligently cluster text segments into coherent topics
        
        Args:
            segments: List of transcript segments with 'text', 'start', 'end' keys
            
        Returns:
            List of topic clusters with metadata
        """
        try:
            # Prepare the text for analysis with preprocessing
            processed_segments = []
            for seg in segments:
                processed_text = self._preprocess_text(seg['text'])
                processed_segments.append(f"[{seg['start']:.2f}-{seg['end']:.2f}s] {processed_text}")
            
            full_text = "\n".join(processed_segments)
            
            # Create the clustering prompt
            system_prompt = """You are an expert content analyst and editor. Your task is to analyze a video transcript and intelligently group the content into coherent topics.

IMPORTANT: The transcript may contain transcription errors, spelling mistakes, and grammatical issues. Your job is to:

1. **Understand the intended meaning** despite transcription errors
2. **Correct spelling and grammatical mistakes** while preserving the original intent
3. **Identify natural topic boundaries** based on content shifts
4. **Create meaningful, descriptive topic names** that reflect the actual content
5. **Generate clean, professional descriptions** free from transcription errors

Content Analysis Guidelines:
- Look beyond surface-level transcription errors to understand the core concepts
- Group related segments that discuss the same subject matter
- Each topic should represent a distinct concept or theme
- Aim for 3-8 topics depending on content length and complexity
- Topic names should be clear, descriptive, and professional

Return your analysis as a JSON object with this exact structure:
{
  "topics": [
    {
      "id": 1,
      "name": "Clear Descriptive Topic Name",
      "description": "Clean, error-free description of what this topic covers",
      "start_time": 0.0,
      "end_time": 120.5,
      "keywords": ["corrected_keyword1", "corrected_keyword2", "corrected_keyword3"],
      "segment_indices": [0, 1, 2, 3]
    }
  ]
}

Make sure the JSON is valid and complete. All text should be grammatically correct and professional."""

            user_prompt = f"""Analyze this video transcript and cluster it into coherent topics. 

TRANSCRIPT (may contain transcription errors):
{full_text}

CRITICAL TRANSCRIPTION ERROR PATTERNS TO FIX:
- Algorithm names: "bubble soar" → "bubble sort", "quick soar" → "quicksort", "tim soar" → "timsort"
- Common errors: "Quicksor" → "Quicksort", "Heapsor" → "Heapsort", "algorythm" → "algorithm"
- Technical terms: "comparsion" → "comparison", "performence" → "performance"

IMPORTANT NOTES:
- This transcript contains many spelling mistakes, especially in algorithm names
- Focus on understanding the intended meaning rather than the exact wording
- ALWAYS correct algorithm and technical term spelling errors
- Create professional, error-free topic names and descriptions  
- Look for natural content boundaries and thematic shifts
- Ensure all technical terminology is spelled correctly in your output

Please provide a comprehensive topic analysis following the specified JSON format with clean, corrected content."""

            data = {
                "model": self.clustering_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,  # Slightly higher for better error correction creativity
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
            
            # Prepare content for summarization with preprocessing
            processed_topic_segments = []
            for seg in topic_segments:
                processed_text = self._preprocess_text(seg['text'])
                processed_topic_segments.append(f"[{seg['start']:.2f}s] {processed_text}")
            
            topic_text = "\n".join(processed_topic_segments)
            
            system_prompt = """You are an expert content editor and summarizer. Your task is to create a comprehensive, polished summary of a specific topic from a video transcript.

CRITICAL INSTRUCTIONS:
1. **Error Correction**: The transcript contains transcription errors, spelling mistakes, and grammatical issues. You MUST:
   - Identify and correct spelling mistakes (e.g., "algorythm" → "algorithm", "comparsion" → "comparison")
   - Fix grammatical errors and awkward phrasing
   - Interpret garbled or unclear text based on context
   - Use proper technical terminology and spelling

2. **Content Enhancement**: 
   - You can ADD explanatory words and phrases to improve clarity
   - Expand abbreviations and clarify technical terms
   - Provide better sentence structure while keeping the original meaning
   - Make the content more accessible to general audiences

3. **Summary Quality**:
   - Write in clear, professional, human-friendly language
   - Use proper grammar, punctuation, and spelling throughout
   - Organize information logically with smooth transitions
   - Keep it engaging and informative (2-4 well-structured paragraphs)
   - Focus on key concepts, explanations, examples, and practical takeaways

4. **Style Guidelines**:
   - Use active voice where possible
   - Write for someone who wants to understand the topic without watching the video
   - Include specific details and examples when available
   - Maintain a friendly, educational tone

Remember: Your goal is to transform potentially error-filled transcript text into polished, professional, human-readable content that accurately conveys the intended meaning."""

            user_prompt = f"""Create a polished, professional summary for the topic "{topic['name']}":

**Topic Information:**
- Description: {topic.get('description', 'N/A')}
- Key Concepts: {', '.join(topic.get('keywords', []))}
- Duration: {topic.get('start_time', 0):.1f}s to {topic.get('end_time', 0):.1f}s

**Raw Transcript Content (contains errors that need correction):**
{topic_text}

**CRITICAL ERROR CORRECTIONS - FIX THESE EXACTLY:**

ALGORITHM NAME ERRORS (ALWAYS FIX):
❌ "bubble soar" → ✅ "bubble sort"
❌ "quick soar" → ✅ "quicksort" 
❌ "Quicksor" → ✅ "Quicksort"
❌ "tim soar" → ✅ "timsort"
❌ "heap soar" → ✅ "heapsort"
❌ "Heapsor" → ✅ "Heapsort"
❌ "merge soar" → ✅ "merge sort"
❌ "insertion soar" → ✅ "insertion sort"
❌ "selection soar" → ✅ "selection sort"

TECHNICAL TERM ERRORS (ALWAYS FIX):
❌ "algorythm/algorythms" → ✅ "algorithm/algorithms"
❌ "comparsion/comparsions" → ✅ "comparison/comparisons"  
❌ "performence" → ✅ "performance"
❌ "efficency" → ✅ "efficiency"
❌ "implemention" → ✅ "implementation"

IMPORTANT: Scan the entire text for these specific errors and fix every instance!

**Your Task:**
1. **FIRST: Scan for and fix all algorithm name errors** (soar→sort, sor→sort, missing letters)
2. **Correct all spelling and grammatical errors** in the transcript
3. **Fix technical terminology** - ensure algorithm names, programming concepts, and technical terms are spelled correctly
4. **Improve clarity and readability** while preserving the original meaning  
5. **Add explanatory context** where helpful for understanding (e.g., explain time complexity, algorithm benefits)
6. **Remove filler words** like "um", "uh", "you know", "like" that don't add value
7. **Create a flowing, well-structured summary** that someone can easily understand
8. **Use proper technical terminology** and professional language
9. **Make it human-friendly** - write as if explaining to an interested friend who wants to learn

**Output Requirements:**
- 2-4 well-written paragraphs
- Perfect spelling, grammar, and punctuation
- Clear, engaging, and informative content
- Professional yet accessible tone
- Include specific examples and details mentioned in the content

Please provide a polished, error-free summary that transforms the raw transcript into professional, readable content."""

            # Add an example to show the model exactly what we want
            example_message = """EXAMPLE OF PROPER ERROR CORRECTION:

Input (with errors): "Quicksor is one of the most popular algorythms that also uses the divide and conqur strategy. Programming languges like JavaScript use Quicksor in their standrd library fetures."

Output (corrected): "Quicksort is one of the most popular algorithms that uses the divide and conquer strategy. This efficient sorting algorithm works by selecting a 'pivot' element and partitioning the array around it. Programming languages like JavaScript use Quicksort in their standard library features because of its excellent average-case performance of O(n log n)."

Notice how I:
1. Fixed "Quicksor" → "Quicksort"
2. Fixed "algorythms" → "algorithms" 
3. Fixed "conqur" → "conquer"
4. Fixed "languges" → "languages"
5. Fixed "standrd" → "standard"
6. Fixed "fetures" → "features"
7. Added explanatory context about how the algorithm works
8. Made the text flow naturally and professionally

Now apply the same approach to the content below."""

            data = {
                "model": self.summary_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example_message},
                    {"role": "assistant", "content": "I understand. I will carefully correct all spelling and grammar errors, especially algorithm names, while preserving the original meaning and adding helpful context. I'll make sure to fix common transcription errors like 'soar' → 'sort', 'sor' → 'sort', and technical term misspellings."},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.6,  # Higher temperature to encourage creative error correction
                "max_tokens": 4000,  # Increased tokens for comprehensive summaries without truncation
                "top_p": 0.9  # Add nucleus sampling for better text generation
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