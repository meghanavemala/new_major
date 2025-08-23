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
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.WARNING)
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

def _should_force_single_topic(prompt_context: str) -> bool:
    """
    Determine if a prompt context indicates that only one topic should be extracted.
    
    Args:
        prompt_context: The prompt context text
        
    Returns:
        bool: True if the prompt indicates a single topic focus, False otherwise
    """
    if not prompt_context or not prompt_context.strip():
        return False
    
    # Convert to lowercase for case-insensitive matching
    prompt_lower = prompt_context.lower().strip()
    
    # Keywords that suggest multiple topics (negations)
    multi_topic_indicators = [
        'both', 'and', 'various', 'different', 'multiple', 'several', 'all', 'aspects', 'approaches', 'concepts'
    ]
    
    # Check if any multi-topic indicator is in the prompt
    for indicator in multi_topic_indicators:
        if indicator in prompt_lower:
            return False
    
    # Keywords that suggest a single topic focus
    single_topic_indicators = [
        # Direct indicators
        'only', 'just', 'single', 'one', 'main', 'primary', 'core', 'central',
        
        # Focus indicators
        'focus on', 'focus only on', 'concentrate on', 'emphasize', 'highlight',
        'specifically', 'particularly', 'exclusively', 'solely',
        
        # Topic limitation indicators
        'limit to', 'restrict to', 'stick to', 'confine to',
        
        # Question formats that imply single topic
        'what is', 'what are', 'explain', 'describe', 'tell me about',
        'how to', 'how do', 'why is', 'why are', 'when is', 'when are'
    ]
    
    # Check if any indicator is in the prompt
    for indicator in single_topic_indicators:
        if indicator in prompt_lower:
            return True
    
    # Special patterns for single topic requests
    special_patterns = [
        r'^.*only.*$',  # Anything with "only"
        r'^.*just.*$',   # Anything with "just"
        r'^.*main.*$',   # Anything with "main"
        r'^explain.*$',  # Starts with "explain"
        r'^describe.*$', # Starts with "describe"
        r'^what.*is.*$', # Question format "what is"
        r'^how.*to.*$',  # Question format "how to"
    ]
    
    import re
    for pattern in special_patterns:
        if re.match(pattern, prompt_lower):
            return True
    
    return False

class TextPreprocessor:
    """Enhanced text preprocessing utilities for clustering with prompt-based optimization."""
    
    def __init__(self, language: str = 'en', prompt_context: Optional[str] = None):
        """Initialize the text preprocessor.
        
        Args:
            language: Language code ('en', 'hi', 'kn')
            prompt_context: Optional prompt context to guide clustering
        """
        self.language = language
        self.prompt_context = prompt_context
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = LANGUAGE_STOPWORDS.get(language, set())
        
        # Enhanced stopwords based on prompt context
        if prompt_context:
            self._enhance_stopwords_from_prompt()
        
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
    
    def _enhance_stopwords_from_prompt(self):
        """Enhance stopwords based on prompt context for better clustering."""
        if not self.prompt_context:
            return
            
        # Extract key concepts from prompt
        prompt_tokens = self.tokenize(self.prompt_context.lower())
        
        # Add prompt-specific terms to stopwords to avoid over-clustering
        prompt_keywords = set()
        for token in prompt_tokens:
            if len(token) > 3 and token not in self.stopwords:
                prompt_keywords.add(token)
        
        # Add these to stopwords to prevent them from dominating clusters
        self.stopwords.update(prompt_keywords)
    
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
    method: str = 'sbert',
    language: str = 'en',
    min_topic_size: int = 2,
    max_features: int = 5000,
    min_cluster_size: int = 2,
    similarity_threshold: float = 0.7,
    prompt_context: Optional[str] = None
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
        
        # Initialize preprocessor with prompt context
        preprocessor = TextPreprocessor(language, prompt_context)
        
        # Preprocess texts
        preprocessed_texts = [preprocessor.preprocess(text) for text in texts]
        
        if method == 'sbert':
            # --- ENHANCED SBERT METHOD ---
            logger.info("Using Enhanced SBERT for topic clustering...")
            from sentence_transformers import SentenceTransformer
            
            # Use better model for improved embeddings
            model = SentenceTransformer('all-mpnet-base-v2')  # Better than MiniLM
            
            # Use raw preprocessed_texts for embeddings
            embeddings = model.encode(preprocessed_texts, convert_to_numpy=True, show_progress_bar=True)
            
            # Enhanced KMeans with better initialization
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=20,  # More initializations for better results
                max_iter=500,  # More iterations
                tol=1e-4  # Tighter tolerance
            )
            topic_assignments = kmeans.fit_predict(embeddings)
            
        elif method == 'enhanced_sbert':
            # --- MOST ADVANCED SBERT METHOD ---
            logger.info("Using Most Advanced SBERT for topic clustering...")
            from sentence_transformers import SentenceTransformer
            
            # Use the best available model
            model = SentenceTransformer('all-mpnet-base-v2')
            
            # Get embeddings
            embeddings = model.encode(preprocessed_texts, convert_to_numpy=True, show_progress_bar=True)
            
            # Use HDBSCAN for better clustering (automatically determines number of clusters)
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            topic_assignments = clusterer.fit_predict(embeddings)
            
            # Handle noise points and ensure we have the right number of clusters
            if -1 in topic_assignments:
                # Assign noise points to nearest clusters
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(embeddings[topic_assignments != -1])
                noise_indices = np.where(topic_assignments == -1)[0]
                if len(noise_indices) > 0:
                    noise_embeddings = embeddings[noise_indices]
                    nearest_clusters = nn.kneighbors(noise_embeddings, return_distance=False)
                    for i, idx in enumerate(noise_indices):
                        topic_assignments[idx] = topic_assignments[nearest_clusters[i][0]]

        # Vectorize texts
        elif method in ['lda', 'nmf']:
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
            if len(cluster) >= min_cluster_size
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

def determine_optimal_clusters(
    segments: List[Dict[str, Any]],
    language: str = 'en',
    prompt_context: Optional[str] = None,
    max_clusters: int = 10
) -> int:
    """Determine optimal number of clusters based on content and prompt context."""
    if not segments:
        return 3
    
    # Check if prompt context indicates single topic focus
    if prompt_context and _should_force_single_topic(prompt_context):
        logger.info("Prompt context indicates single topic focus. Forcing single cluster.")
        return 1
    
    # Extract texts
    texts = [seg.get('text', '') for seg in segments]
    
    # Initialize preprocessor with prompt context
    preprocessor = TextPreprocessor(language, prompt_context)
    preprocessed_texts = [preprocessor.preprocess(text) for text in texts]
    
    # Use SBERT for embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(preprocessed_texts, convert_to_numpy=True)
    
    # Try different numbers of clusters and evaluate
    best_score = -1
    best_n_clusters = 3
    
    for n_clusters in range(2, min(max_clusters + 1, len(texts) // 2)):
        try:
            # Use KMeans with silhouette score
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            if len(set(cluster_labels)) > 1:  # Need at least 2 clusters
                score = silhouette_score(embeddings, cluster_labels)
                
                # Bonus for prompt-aligned clustering
                prompt_bonus = 0
                if prompt_context:
                    # Check if clusters align with prompt themes
                    prompt_bonus = _calculate_prompt_alignment_bonus(
                        embeddings, cluster_labels, prompt_context, model
                    )
                
                final_score = score + prompt_bonus
                
                if final_score > best_score:
                    best_score = final_score
                    best_n_clusters = n_clusters
                    
        except Exception as e:
            logger.warning(f"Error evaluating {n_clusters} clusters: {e}")
            continue
    
    logger.info(f"Optimal number of clusters: {best_n_clusters} (score: {best_score:.3f})")
    return best_n_clusters

def _calculate_prompt_alignment_bonus(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    prompt_context: Optional[str],
    model: SentenceTransformer
) -> float:
    """Calculate bonus score for prompt-aligned clustering."""
    try:
        # Check if prompt_context is None
        if not prompt_context:
            return 0.0
            
        # Get prompt embedding
        prompt_embedding = model.encode([prompt_context], convert_to_numpy=True)[0]
        
        # Calculate cluster centroids
        unique_labels = set(cluster_labels)
        centroids = []
        
        for label in unique_labels:
            cluster_embeddings = embeddings[cluster_labels == label]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids.append(centroid)
        
        # Calculate similarity between prompt and cluster centroids
        similarities = []
        for centroid in centroids:
            similarity = np.dot(prompt_embedding, centroid) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(centroid)
            )
            similarities.append(similarity)
        
        # Return average similarity as bonus
        return float(np.mean(similarities) * 0.1)  # Small bonus to avoid overwhelming silhouette score
        
    except Exception as e:
        logger.warning(f"Error calculating prompt alignment bonus: {e}")
        return 0.0

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
        
        return [str(feature_names[i]) for i in top_indices if i < len(feature_names)]
    
    except Exception as e:
        logger.error(f"Error in extract_keywords: {e}", exc_info=True)
        return []
