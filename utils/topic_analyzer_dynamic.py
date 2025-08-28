import logging
import traceback
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.neighbors import NearestNeighbors
import yake
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Cached embedder
_EMBEDDER: Optional[SentenceTransformer] = None

def get_embedder() -> SentenceTransformer:
    """Return a cached sentence transformer instance."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-mpnet-base-v2")
    return _EMBEDDER

def preprocess_text(text: str, language: str = 'en') -> str:
    """Clean and preprocess text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        # Basic cleaning
        text = text.lower().strip()
        
        # Remove URLs and social media handles
        text = re.sub(r'https?://\S+|www\.\S+|@\w+|#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Handle special characters based on language
        if language == 'en':
            # For English, keep basic punctuation that might be meaningful
            text = re.sub(r'[^\w\s.,!?]', ' ', text)
        else:
            # For other languages, be more conservative with punctuation removal
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove very short segments
        if len(text.split()) < 3:
            return ""
            
        return text
    except Exception as e:
        logger.error(f"Error in preprocess_text: {str(e)}")
        return text  # Return original text if processing fails

def analyze_topic_segments_dynamic(
    segments: List[Dict[str, Any]],
    language: str = 'en',
    user_prompt: Optional[str] = None,
    min_topic_size: int = 2,
    similarity_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Analyze video segments and identify topics using dynamic clustering.
    
    Args:
        segments: List of transcript segments
        language: Language code
        user_prompt: Optional context from user query
        min_topic_size: Minimum segments per topic
        similarity_threshold: Similarity threshold for clustering
        
    Returns:
        List of topics with their segments and metadata
    """
    if not segments:
        return []

    try:
        # Extract and validate text content
        texts = []
        valid_segments = []
        
        for seg in segments:
            text = seg.get('text', '').strip()
            if text and len(text.split()) >= 3:  # Only include segments with meaningful content
                texts.append(text)
                valid_segments.append(seg)
        
        if not texts:
            logger.error("No valid text segments found")
            return []
            
        # Preprocess texts
        preprocessed_texts = [preprocess_text(text, language) for text in texts]
        preprocessed_texts = [text for text in preprocessed_texts if text.strip()]  # Remove empty results
        
        if not preprocessed_texts:
            logger.error("No valid preprocessed texts remaining")
            return []
        
        # Get embeddings using SentenceTransformer
        try:
            model = get_embedder()
            embeddings = model.encode(preprocessed_texts, convert_to_numpy=True, show_progress_bar=True)
        except Exception as e:
            logger.error(f"Error in embedding generation: {str(e)}")
            return []
            
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Adjust HDBSCAN parameters based on content
        min_cluster_size = max(min_topic_size, 2)  # At least 2 segments per cluster
        min_samples = 1 if len(preprocessed_texts) < 10 else 2  # Adjust based on data size
        
        # Use HDBSCAN for dynamic clustering with optimized parameters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            cluster_selection_epsilon=0.5  # More lenient cluster selection
        )
        
        # Get cluster assignments
        topic_assignments = clusterer.fit_predict(embeddings)
        
        # Handle noise points (-1 labels) more robustly
        if -1 in topic_assignments and len(set(topic_assignments)) > 1:
            core_mask = topic_assignments != -1
            if np.any(core_mask):
                try:
                    # Use similarity matrix to assign noise points to nearest clusters
                    noise_indices = np.where(topic_assignments == -1)[0]
                    core_indices = np.where(core_mask)[0]
                    
                    for noise_idx in noise_indices:
                        # Get similarities to all core points
                        similarities = similarity_matrix[noise_idx][core_indices]
                        most_similar_idx = core_indices[np.argmax(similarities)]
                        topic_assignments[noise_idx] = topic_assignments[most_similar_idx]
                except Exception as e:
                    logger.warning(f"Error in noise point assignment: {str(e)}")
                    # Keep as noise points if assignment fails
        
        # Extract keywords for each topic
        topics = []
        unique_topics = set(topic_assignments)
        
        for topic_idx in unique_topics:
            if topic_idx == -1:
                continue
                
            # Get segments for this topic
            topic_segment_indices = [i for i, t in enumerate(topic_assignments) if t == topic_idx]
            topic_segments = [segments[i] for i in topic_segment_indices]
            
            if len(topic_segments) < min_topic_size:
                continue
            
            # Extract keywords using YAKE
            topic_text = " ".join([seg.get('text', '') for seg in topic_segments])
            kw_extractor = yake.KeywordExtractor(
                lan=language,
                n=2,  # ngram size
                dedupLim=0.3,
                top=5,
                features=None
            )
            keywords = [kw for kw, _ in kw_extractor.extract_keywords(topic_text)]
            
            # Create topic entry with Python native types (not numpy types)
            topic_entry = {
                'id': int(topic_idx),  # Convert numpy.int64 to Python int
                'name': f"Topic {int(topic_idx) + 1}: {keywords[0] if keywords else ''}",
                'keywords': keywords,
                'segments': topic_segments,
                'start_time': float(min(seg.get('start', 0) for seg in topic_segments)),
                'end_time': float(max(seg.get('end', 0) for seg in topic_segments))
            }
            
            # Add topic ID to original segments
            for i in topic_segment_indices:
                segments[i]['topic_id'] = topic_idx
            
            topics.append(topic_entry)
        
        return topics
        
    except Exception as e:
        logger.error(f"Error in analyze_topic_segments_dynamic: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []
