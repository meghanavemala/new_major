import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Language-specific settings
LANGUAGE_STOPWORDS = {
    'en': set(stopwords.words('english')),
    'hi': set(stopwords.words('hindi') if 'hindi' in stopwords.fileids() else []),
    'kn': set(stopwords.words('kannada') if 'kannada' in stopwords.fileids() else [])
}

# Add custom stopwords for each language
CUSTOM_STOPWORDS = {
    'en': {'like', 'one', 'would', 'get', 'also', 'could', 'may', 'even', 'much', 'many', 'well'},
    'hi': set(),
    'kn': set()
}

# Combine NLTK and custom stopwords
for lang in LANGUAGE_STOPWORDS:
    LANGUAGE_STOPWORDS[lang].update(CUSTOM_STOPWORDS.get(lang, set()))

class TextPreprocessor:
    """Text preprocessing utilities for clustering."""
    
    def __init__(self, language: str = 'en'):
        """Initialize the text preprocessor.
        
        Args:
            language: Language code ('en', 'hi', 'kn')
        """
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = LANGUAGE_STOPWORDS.get(language, set())
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation (language-specific handling)
        if self.language == 'en':
            # Keep sentence boundaries for English
            text = re.sub(r'[^\w\s.!?]', '', text)
        else:
            # For other languages, be more conservative with punctuation removal
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if not text.strip():
            return []
            
        # Tokenize
        tokens = word_tokenize(text, language='english' if self.language == 'en' else self.language)
        
        # Remove stopwords and short tokens
        tokens = [
            token for token in tokens 
            if token not in self.stopwords 
            and len(token) > 2
            and not token.isdigit()
        ]
        
        # Lemmatization (for English only)
        if self.language == 'en':
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return ' '.join(tokens)

def cluster_segments(
    segments: List[Dict[str, Any]],
    n_clusters: int = 5,
    method: str = 'lda',
    language: str = 'en',
    min_topic_size: int = 2,
    max_features: int = 5000
) -> List[List[Dict[str, Any]]]:
    """Cluster text segments into topics.
    
    Args:
        segments: List of text segments with 'text' keys
        n_clusters: Number of clusters/topics to create
        method: Clustering method ('lda', 'nmf', 'kmeans', 'dbscan')
        language: Language code ('en', 'hi', 'kn')
        min_topic_size: Minimum number of segments per topic
        max_features: Maximum number of features for vectorization
        
    Returns:
        List of clusters, where each cluster is a list of segments
    """
    if not segments:
        return []
    
    # Validate language
    language = language.lower()
    if language not in ['en', 'hi', 'kn']:
        logger.warning(f"Unsupported language: {language}. Defaulting to English.")
        language = 'en'
    
    try:
        # Extract texts from segments
        texts = [seg.get('text', '') for seg in segments]
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor(language)
        
        # Preprocess texts
        preprocessed_texts = [preprocessor.preprocess(text) for text in texts]
        
        # Vectorize texts
        if method in ['lda', 'nmf']:
            # For topic modeling, use count vectorizer
            vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words=list(LANGUAGE_STOPWORDS[language]) if language in LANGUAGE_STOPWORDS else None
            )
            X = vectorizer.fit_transform(preprocessed_texts)
            
            # Apply topic modeling
            if method == 'lda':
                model = LatentDirichletAllocation(
                    n_components=n_clusters,
                    random_state=42,
                    learning_method='online',
                    max_iter=10
                )
                topic_assignments = model.fit_transform(X).argmax(axis=1)
            else:  # NMF
                model = NMF(
                    n_components=n_clusters,
                    random_state=42,
                    max_iter=1000
                )
                topic_assignments = model.fit_transform(X).argmax(axis=1)
                
        else:  # kmeans or dbscan
            # For clustering, use TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words=list(LANGUAGE_STOPWORDS[language]) if language in LANGUAGE_STOPWORDS else None
            )
            X = vectorizer.fit_transform(preprocessed_texts)
            
            if method == 'kmeans':
                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10
                )
                topic_assignments = model.fit_predict(X)
            else:  # dbscan
                model = DBSCAN(
                    eps=0.5,
                    min_samples=2,
                    metric='cosine'
                )
                topic_assignments = model.fit_predict(X)
                
                # Handle noise points (assigned to -1)
                if -1 in topic_assignments:
                    # Assign noise points to their own clusters
                    max_cluster = max(topic_assignments) + 1
                    topic_assignments = [x if x != -1 else max_cluster + i for i, x in enumerate(topic_assignments)]
        
        # Assign cluster labels to segments
        for i, seg in enumerate(segments):
            seg['cluster'] = int(topic_assignments[i])
        
        # Group segments by cluster
        clusters = defaultdict(list)
        for seg in segments:
            clusters[seg['cluster']].append(seg)
        
        # Filter small clusters
        filtered_clusters = [
            cluster for cluster in clusters.values() 
            if len(cluster) >= min_topic_size
        ]
        
        # Sort clusters by size (largest first)
        filtered_clusters.sort(key=len, reverse=True)
        
        # Limit to n_clusters if we have more
        if len(filtered_clusters) > n_clusters:
            filtered_clusters = filtered_clusters[:n_clusters]
        
        logger.info(f"Created {len(filtered_clusters)} clusters with {sum(len(c) for c in filtered_clusters)} segments")
        
        return filtered_clusters
        
    except Exception as e:
        logger.error(f"Error in cluster_segments: {e}", exc_info=True)
        
        # Fallback: return all segments in a single cluster
        logger.warning("Falling back to single cluster")
        return [segments]

def extract_keywords(
    texts: List[str],
    n_keywords: int = 5,
    language: str = 'en',
    max_features: int = 5000
) -> List[str]:
    """Extract keywords from a list of texts."""
    if not texts:
        return []
    
    try:
        # Preprocess texts
        preprocessor = TextPreprocessor(language)
        preprocessed_texts = [' '.join(preprocessor.tokenize(text)) for text in texts]
        
        # Vectorize using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=list(LANGUAGE_STOPWORDS[language]) if language in LANGUAGE_STOPWORDS else None
        )
        X = vectorizer.fit_transform(preprocessed_texts)
        
        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate average TF-IDF score for each word across all documents
        avg_tfidf = X.mean(axis=0).A1
        
        # Sort words by average TF-IDF score
        top_indices = avg_tfidf.argsort()[-n_keywords:][::-1]
        
        # Extract top keywords
        keywords = [feature_names[i] for i in top_indices if i < len(feature_names)]
        
        return keywords
    
    except Exception as e:
        logger.error(f"Error in extract_keywords: {e}", exc_info=True)
        return []
