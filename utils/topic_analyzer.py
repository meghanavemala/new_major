"""
Topic Analyzer Module

This module provides advanced topic analysis for video transcripts:
1. Dynamic topic extraction based on content
2. Topic naming based on keywords and context
3. Timestamp-based segmentation for better keyframe selection
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import re
from collections import defaultdict
import spacy
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS
import yake
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')

# Load spaCy model for NER and topic extraction
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    logger.warning("spaCy model not available. Some features will be limited.")
    SPACY_AVAILABLE = False

"""
Performance optimizations
-------------------------
1) Reuse a single SentenceTransformer instance across calls (embedding is expensive).
2) Compute embeddings once and pass them through to clustering and k-selection.
3) Provide a fast heuristic for selecting number of clusters to avoid silhouette loops.
4) Use MiniBatchKMeans for faster clustering on larger inputs.
5) Skip spaCy NER by default in fast mode to avoid heavy startup cost.
"""

# Cached embedder (initialized on demand)
_EMBEDDER: Optional[SentenceTransformer] = None

def get_embedder() -> SentenceTransformer:
    """Return a cached multilingual embedder instance."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _EMBEDDER

# Common conversational fillers to be removed
EXTENDED_FILLERS = {
    'oh', 'uh', 'smth', 'hmm', 'okay', 'right', 'just', 'actually', 'yeah', 'um', 'so', 'like',
    'well', 'you know', 'i mean', 'kind of', 'sort of', 'basically', 'literally', 'really',
    'actually', 'seriously', 'honestly', 'basically', 'totally', 'absolutely', 'maybe', 'perhaps',
    'probably', 'obviously', 'apparently', 'basically', 'like i said', 'you see', 'i guess'
}

# Language-specific settings
LANGUAGE_STOPWORDS = {
    'en': set(stopwords.words('english')),
    'hi': set(stopwords.words('hindi') if 'hindi' in stopwords.fileids() else []),
    'kn': set(stopwords.words('kannada') if 'kannada' in stopwords.fileids() else []),
    'te': set(),  # Telugu
    'ta': set(),  # Tamil
    'ml': set(),  # Malayalam
    'mr': set(),  # Marathi
    'ur': set()   # Urdu
}

# Add custom stopwords for each language
CUSTOM_STOPWORDS = {
    'en': {'like', 'one', 'would', 'get', 'also', 'could', 'may', 'even', 'much', 'many', 'well', 
           'thing', 'things', 'way', 'something', 'everything', 'anything', 'nothing', 'say', 'said',
           'know', 'going', 'go', 'goes', 'went', 'gone', 'just', 'good', 'right', 'now', 'see', 'come'},
    'hi': {'तो', 'है', 'और', 'में', 'की', 'हैं', 'हैं', 'है', 'नहीं', 'हैं', 'कर', 'की', 'मैं', 'हूं', 'है', 'हैं', 'है', 'हैं'},
    'kn': {'ಮತ್ತು', 'ನಾನು', 'ನನ್ನ', 'ಅದು', 'ಅವರು', 'ಇದು', 'ಈ', 'ಅವರ', 'ಅದನ್ನು', 'ಅದರ', 'ಅವರನ್ನು'},
    'te': {'మరియు', 'నేను', 'నా', 'అది', 'అతను', 'ఆమె', 'మనం', 'మీరు', 'వారు', 'ఇది', 'ఆ', 'అదే', 'కాదు'},
    'ta': {'மற்றும்', 'நான்', 'என்', 'அது', 'அவர்', 'இது', 'இந்த', 'ஒரு', 'நாம்', 'நீங்கள்', 'அவர்கள்'},
    'ml': {'ഞാൻ', 'എന്റെ', 'നീ', 'നിന്റെ', 'അവൻ', 'അവൾ', 'അത്', 'ഇത്', 'നമ്മൾ', 'നിങ്ങൾ', 'അവർ'},
    'mr': {'आणि', 'मी', 'माझं', 'तो', 'ती', 'ते', 'हा', 'ही', 'हे', 'आम्ही', 'तुम्ही', 'ते', 'होत', 'आहे'},
    'ur': {'اور', 'میں', 'نے', 'کی', 'ہے', 'ہیں', 'ہے', 'اور', 'ہے', 'ہیں', 'ہے', 'تھا', 'تھی', 'تھے'}
}

# Combine NLTK, custom stopwords, and extended fillers for each language
for lang in LANGUAGE_STOPWORDS:
    LANGUAGE_STOPWORDS[lang].update(CUSTOM_STOPWORDS.get(lang, set()))
    LANGUAGE_STOPWORDS[lang].update(EXTENDED_FILLERS)


def preprocess_text(text: str, language: str = 'en') -> str:
    """
    Clean and preprocess text for analysis.
    
    Args:
        text: Input text to preprocess
        language: Language code (en, hi, kn, te, ta, ml, mr, ur)
        
    Returns:
        Preprocessed text with filler words, stopwords, and noise removed
    """
    if not isinstance(text, str) or not text.strip():
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and social media handles
    text = re.sub(r'https?:\/\/\S+|www\.\S+|@\w+|#\w+', '', text)
    
    # Remove HTML tags and special characters
    text = re.sub(r'<.*?>|&[a-z]+;', '', text)
    
    # Handle language-specific preprocessing
    if language in ['en']:
        # Keep sentence boundaries for English
        text = re.sub(r'[^\w\s.!?]', '', text)
    else:
        # For Indian languages, be more conservative with punctuation removal
        # Preserve common punctuation used in these languages
        text = re.sub(r'[^\w\s\u0900-\u097F\u0C80-\u0CFF\u0980-\u09FF\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF]', ' ', text)
    
    # Remove extra whitespace and normalize spaces
    text = ' '.join(text.split())
    
    return text


def tokenize_text(text: str, language: str = 'en') -> List[str]:
    """
    Tokenize text into words with advanced stopword and filler word removal.
    
    Args:
        text: Input text to tokenize
        language: Language code (en, hi, kn, te, ta, ml, mr, ur)
        
    Returns:
        List of cleaned tokens with stopwords and fillers removed
    """
    if not text.strip():
        return []
    
    # Get language-specific stopwords and fillers
    stopwords_set = LANGUAGE_STOPWORDS.get(language, set())
    
    # Tokenize based on language
    try:
        if language == 'en':
            tokens = nltk.word_tokenize(text, language='english')
        else:
            # For Indian languages, use a more generic tokenizer
            tokens = text.split()
    except:
        # Fallback to simple whitespace tokenization
        tokens = text.split()
    
    # Clean and filter tokens
    cleaned_tokens = []
    for token in tokens:
        # Skip short tokens, numbers, and stopwords
        if (len(token) <= 2 or 
            token.isdigit() or 
            token in stopwords_set or
            token in string.punctuation or
            token in EXTENDED_FILLERS):
            continue
            
        # Additional cleaning for each token
        token = token.strip('\'"`.,!?;:()[]{}-_+=\\|/<>~@#$%^&*')
        
        if token and len(token) > 2:
            cleaned_tokens.append(token.lower())
    
    return cleaned_tokens


def determine_optimal_clusters(
    texts: List[str],
    max_clusters: int = 10,
    min_clusters: int = 2,
    language: str = 'en',
    use_elbow: bool = False,
    embeddings: Optional[np.ndarray] = None,
    strategy: str = 'auto'
) -> int:
    """
    Determine the optimal number of clusters using silhouette score with multilingual embeddings.
    
    Args:
        texts: List of text segments to cluster
        max_clusters: Maximum number of clusters to consider
        min_clusters: Minimum number of clusters to consider
        language: Language code for text processing
        use_elbow: If True, use elbow method with inertia instead of silhouette score
        
    Returns:
        Optimal number of clusters based on the highest silhouette score or elbow method
    """
    if len(texts) < min_clusters:
        return max(1, len(texts))
    
    # Fast heuristic path: choose k ~ sqrt(N), clamped
    n_samples = len(texts)
    if strategy in ('fast', 'heuristic'):
        if n_samples <= 1:
            return 1
        if n_samples == 2:
            return 2
        k_guess = int(np.ceil(np.sqrt(n_samples)))
        return max(min_clusters, min(max_clusters, k_guess))

    logger.info("Generating multilingual embeddings for clustering...")
    try:
        # Generate embeddings for all texts if not provided
        if embeddings is None:
            embedder = get_embedder()
            embeddings = embedder.encode(texts, show_progress_bar=False)
        
        # Normalize embeddings for better clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)
        
        # Try different numbers of clusters
        best_score = -1
        best_n_clusters = min_clusters
        max_clusters = min(max_clusters, len(texts) - 1)
        
        # Store metrics for analysis
        metrics = []
        
        # Evaluate a small, targeted set of k values to reduce runtime
        candidate_ks: List[int] = []
        # Always consider small ks
        candidate_ks.extend([2, 3, 4, 5])
        # Add heuristic-based k
        candidate_ks.append(int(np.ceil(np.sqrt(n_samples))))
        # Unique, in-range, sorted
        candidate_ks = sorted({k for k in candidate_ks if min_clusters <= k <= min(max_clusters, n_samples - 1)})

        for n_clusters in candidate_ks:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings_normalized)
                
                # Calculate silhouette score (higher is better)
                if len(set(labels)) > 1:  # Need at least 2 unique labels
                    if use_elbow:
                        # Use inertia (sum of squared distances to centroids)
                        score = -kmeans.inertia_  # Negative because we're maximizing
                        metric_name = "inertia"
                    else:
                        # Use silhouette score (default)
                        score = silhouette_score(embeddings_normalized, labels)
                        metric_name = "silhouette"
                    
                    metrics.append((n_clusters, score))
                    logger.debug(f"Clusters: {n_clusters}, {metric_name}: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        
            except Exception as e:
                logger.warning(f"Error in clustering with {n_clusters} clusters: {e}")
        
        # If using elbow method, find the "elbow" point
        if use_elbow and len(metrics) > 1:
            from kneed import KneeLocator
            try:
                n_clusters_values = [m[0] for m in metrics]
                scores = [m[1] for m in metrics]
                
                # Find the elbow point (point of maximum curvature)
                kneedle = KneeLocator(
                    n_clusters_values, 
                    scores, 
                    curve='convex', 
                    direction='increasing'
                )
                
                if kneedle.elbow is not None:
                    best_n_clusters = kneedle.elbow
                    logger.info(f"Elbow method selected {best_n_clusters} clusters")
                
            except Exception as e:
                logger.warning(f"Error in elbow method, using best silhouette score: {e}")
        
        logger.info(f"Optimal number of clusters: {best_n_clusters} (best score: {best_score:.3f})")
        return best_n_clusters
        
    except Exception as e:
        logger.error(f"Error in embedding-based clustering, falling back to TF-IDF: {e}")
        # Fallback to TF-IDF if embedding fails
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts)
        
        best_score = -1
        best_n_clusters = min_clusters
        
        candidate_ks: List[int] = []
        candidate_ks.extend([2, 3, 4, 5])
        candidate_ks.append(int(np.ceil(np.sqrt(n_samples))))
        candidate_ks = sorted({k for k in candidate_ks if min_clusters <= k <= min(max_clusters, n_samples - 1)})

        for n_clusters in candidate_ks:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(X, labels)
                    logger.debug(f"TF-IDF Clusters: {n_clusters}, Silhouette: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            except Exception as e:
                logger.warning(f"Error in TF-IDF clustering with {n_clusters} clusters: {e}")
        
        logger.info(f"Using TF-IDF fallback. Optimal clusters: {best_n_clusters} (score: {best_score:.3f})")
        return best_n_clusters


def extract_named_entities(texts: List[str]) -> List[str]:
    """
    Extract named entities from texts using spaCy.
    
    Args:
        texts: List of text segments
        
    Returns:
        List of named entities
    """
    if not SPACY_AVAILABLE:
        return []
    
    entities = []
    combined_text = " ".join(texts[:20])  # Limit to first 20 segments for performance
    
    try:
        doc = nlp(combined_text)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'LOC', 'PRODUCT', 'EVENT']:
                entities.append(ent.text)
    except Exception as e:
        logger.error(f"Error extracting named entities: {e}")
    
    # Count frequencies and return top entities
    entity_counts = defaultdict(int)
    for entity in entities:
        entity_counts[entity] += 1
    
    # Return top 10 entities
    return [entity for entity, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]]


def extract_topic_keywords(texts: List[str], num_keywords: int = 5, language: str = 'en') -> List[str]:
    """
    Extract keywords that best represent a topic using YAKE.
    
    Args:
        texts: List of text segments
        num_keywords: Number of keywords to extract
        language: Language code
        
    Returns:
        List of keywords
    """
    try:
        # Combine texts
        combined_text = " ".join(texts)
        
        # Use YAKE for keyword extraction
        language_code = 'en' if language not in ['en', 'hi', 'kn'] else language
        kw_extractor = yake.KeywordExtractor(
            lan=language_code, 
            n=2,  # Extract 1-2 word keywords
            dedupLim=0.7,  # Avoid similar keywords
            dedupFunc='seqm',
            windowsSize=2,
            top=num_keywords
        )
        
        keywords = kw_extractor.extract_keywords(combined_text)
        
        # Return just the keywords (not scores)
        return [kw[0] for kw in keywords]
    except Exception as e:
        logger.error(f"Error extracting keywords with YAKE: {e}")
        
        # Fallback to simple word frequency
        word_freq = defaultdict(int)
        for text in texts:
            for word in tokenize_text(text, language):
                word_freq[word] += 1
        
        # Return top words
        return [word for word, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:num_keywords]]


def should_force_single_topic(user_prompt: str) -> bool:
    """
    Determine if a user prompt indicates that only one topic should be extracted.
    
    Args:
        user_prompt: The user's prompt text
        
    Returns:
        bool: True if the prompt indicates a single topic focus, False otherwise
    """
    if not user_prompt or not user_prompt.strip():
        return False
    
    # Convert to lowercase for case-insensitive matching
    prompt_lower = user_prompt.lower().strip()
    
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


def generate_topic_name(keywords: List[str], entities: List[str] = None, user_prompt: str = None) -> str:
    """
    Generate a descriptive name for a topic based on keywords and entities.
    Prioritizes named entities and falls back to top keywords if no good entity is found.
    
    Args:
        keywords: List of keywords for the topic, ordered by importance
        entities: List of named entities (e.g., from spaCy NER)
        user_prompt: Optional user prompt to guide topic naming
        
    Returns:
        str: A descriptive topic name using entities or keywords
    """
    if not keywords and not entities:
        return "General Topic"
    
    # Clean and prepare the inputs
    keywords = [k for k in keywords if k] if keywords else []
    entities = [e for e in entities if e] if entities else []
    
    # If user prompt is provided, try to incorporate it into the topic name
    if user_prompt and user_prompt.strip():
        # Extract key terms from user prompt
        prompt_words = user_prompt.lower().split()
        prompt_keywords = [word for word in prompt_words if len(word) > 3 and word not in ['focus', 'highlight', 'emphasize', 'cover', 'main', 'key', 'important', 'specific', 'aspects', 'topics', 'concepts', 'applications', 'examples', 'arguments', 'conclusions', 'insights']]
        
        # Look for prompt keywords in the topic keywords
        for prompt_kw in prompt_keywords[:3]:  # Check top 3 prompt keywords
            for kw in keywords[:5]:  # Check top 5 topic keywords
                if prompt_kw in kw.lower() or kw.lower() in prompt_kw:
                    # Found a match, use this to create a more focused topic name
                    return f"{kw.title()} ({prompt_kw.title()})"
    
    # 1. First try to find a named entity that overlaps with top keywords
    if entities:
        # Create a set of lowercased keywords for faster lookup
        keyword_set = {k.lower() for k in keywords[:5]}  # Only check top 5 keywords
        
        # Look for entity that contains or is contained by a keyword
        for entity in entities:
            entity_lower = entity.lower()
            
            # Check if entity is a significant part of any keyword
            for kw in keywords[:3]:  # Only check top 3 keywords
                kw_lower = kw.lower()
                if (entity_lower in kw_lower or kw_lower in entity_lower) and len(entity_lower) > 2:
                    # Found a good match, use the entity as is (with original capitalization)
                    return entity
            
            # Check if any word in the entity is a keyword
            entity_words = set(entity_lower.split())
            if keyword_set.intersection(entity_words):
                return entity
    
    # 2. If we have entities but no good match, use the most relevant entity
    if entities:
        # Return the first entity that's not too long
        for entity in entities:
            if len(entity.split()) <= 3:  # Prefer shorter entity names
                return entity
        return entities[0]  # Fallback to first entity if all are long
    
    # 3. Fall back to using top keywords
    if not keywords:
        return "General Topic"
        
    # Clean and format keywords
    keywords = [k.strip() for k in keywords if k.strip()]
    
    # For 1-2 keywords, just return them capitalized
    if len(keywords) == 1:
        return keywords[0].title()
    elif len(keywords) == 2:
        return f"{keywords[0].title()} & {keywords[1].title()}"
    
    # For 3+ keywords, use the top 2-3 most significant ones
    significant_keywords = []
    seen_terms = set()
    
    # Add keywords, avoiding duplicates and substrings
    for kw in keywords:
        kw_lower = kw.lower()
        is_substring = any(kw_lower in seen_kw for seen_kw in seen_terms)
        
        if not is_substring and kw_lower not in seen_terms:
            significant_keywords.append(kw)
            seen_terms.add(kw_lower)
            
            if len(significant_keywords) >= 3:  # Limit to top 3 keywords
                break
    
    # Format the final topic name
    if len(significant_keywords) == 1:
        return significant_keywords[0].title()
    elif len(significant_keywords) == 2:
        return f"{significant_keywords[0].title()} & {significant_keywords[1].title()}"
    else:
        return f"{significant_keywords[0].title()}, {significant_keywords[1].title()} & {significant_keywords[2].title()}"


def analyze_topic_segments(
    segments: List[Dict[str, Any]],
    language: str = 'en',
    user_prompt: str = None,
    fast_mode: bool = True
) -> List[Dict[str, Any]]:
    """
    Analyze segments to identify distinct topics with meaningful names using multilingual embeddings.
    
    Args:
        segments: List of transcript segments with 'text' and timestamp fields
        language: Language code for text processing
        user_prompt: Optional user prompt to guide topic clustering and analysis
        
    Returns:
        List of topic dictionaries with name, keywords, start/end times
    """
    if not segments:
        return []
        
    # Extract text from segments
    texts = [segment['text'] for segment in segments]
    
    # If user prompt is provided: in fast mode, avoid polluting all embeddings.
    # Use prompt later for naming; only augment texts when not in fast mode.
    if user_prompt and user_prompt.strip():
        logger.info(f"User prompt detected ({'fast' if fast_mode else 'full'} mode).")
        if not fast_mode:
            enhanced_texts = []
            for text in texts:
                enhanced_text = f"{text} [Context: {user_prompt}]"
                enhanced_texts.append(enhanced_text)
            texts = enhanced_texts
    
    # Generate embeddings once (if we choose embedding-based path)
    embeddings = None
    try:
        embedder = get_embedder()
        embeddings = embedder.encode(texts, show_progress_bar=False)
    except Exception as e:
        logger.error(f"Error generating embeddings, will fall back to TF-IDF later if needed: {e}")

    # Determine optimal number of clusters (fast heuristic by default)
    n_topics = determine_optimal_clusters(
        texts,
        max_clusters=min(10, len(texts)),
        min_clusters=2,
        language=language,
        embeddings=embeddings,
        strategy='heuristic' if fast_mode else 'auto'
    )
    
    # Check if user prompt indicates single topic focus
    if user_prompt and should_force_single_topic(user_prompt):
        logger.info("User prompt indicates single topic focus. Forcing single cluster.")
        n_topics = 1
    
    logger.info(f"Clustering {len(segments)} segments into {n_topics} topics...")
    
    try:
        # Use enhanced clustering
        from .enhanced_clustering import EnhancedClusterAnalyzer
        
        logger.info(f"Using enhanced clustering for {len(segments)} segments...")
        analyzer = EnhancedClusterAnalyzer()
        enhanced_clusters = analyzer.cluster_segments(segments)
        
        if not enhanced_clusters:
            logger.warning("Enhanced clustering returned no clusters, falling back to simpler method")
            raise RuntimeError("No clusters found")
            
        # Convert enhanced clusters to topic format
        topics = []
        for topic_id, cluster in enumerate(enhanced_clusters):
            # Get all text for this topic
            topic_texts = [s['text'] for s in cluster]
            
            # Extract keywords and entities
            keywords = extract_topic_keywords(topic_texts, language=language)
            entities = [] if fast_mode else extract_named_entities(topic_texts)
            
            # Generate topic name
            topic_name = generate_topic_name(keywords, entities, user_prompt)
            
            # Get time range
            start_time = min(s['start'] for s in cluster)
            end_time = max(s['end'] for s in cluster)
            
            # Store cluster information
            for s in cluster:
                s['topic_id'] = topic_id
                
            topics.append({
                'id': topic_id,
                'name': topic_name,
                'keywords': keywords,
                'entities': entities,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'segment_count': len(cluster),
                'embedding_used': 'deberta',
                'coherence_score': analyzer.analyze_cluster_coherence(cluster)
            })
            
    except Exception as e:
        logger.error(f"Error in enhanced clustering, falling back to TF-IDF: {e}")
        # Fallback to TF-IDF if enhanced clustering fails
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts)
        kmeans = MiniBatchKMeans(n_clusters=n_topics, random_state=42, n_init=10, batch_size=max(32, n_topics * 8))
        cluster_labels = kmeans.fit_predict(X)
        
        # Process clusters as before
        for i, segment in enumerate(segments):
            segment['topic_id'] = int(cluster_labels[i])
        
        topics = []
        for topic_id in range(n_topics):
            topic_segments = [s for s in segments if s['topic_id'] == topic_id]
            if not topic_segments:
                continue
            
            topic_texts = [s['text'] for s in topic_segments]
            keywords = extract_topic_keywords(topic_texts, language=language)
            entities = [] if fast_mode else extract_named_entities(topic_texts)
            topic_name = generate_topic_name(keywords, entities, user_prompt)
            
            start_time = min(s['start'] for s in topic_segments)
            end_time = max(s['end'] for s in topic_segments)
            
            topics.append({
                'id': topic_id,
                'name': topic_name,
                'keywords': keywords,
                'entities': entities,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'segment_count': len(topic_segments),
                'embedding_used': 'tfidf'
            })
    
    return topics
    
    logger.info(f"Identified {len(topics)} distinct topics")
    return topics


def get_keyframes_for_topic(keyframes: List[Dict[str, Any]], start_time: float, end_time: float) -> List[Dict[str, Any]]:
    """
    Select keyframes that fall within a topic's time range.
    
    Args:
        keyframes: List of keyframe metadata dictionaries
        start_time: Topic start time in seconds
        end_time: Topic end time in seconds
        
    Returns:
        List of keyframes within the topic time range
    """
    # Filter keyframes by timestamp
    topic_keyframes = []
    
    for kf in keyframes:
        timestamp = kf.get('timestamp')
        if timestamp is not None and start_time <= timestamp <= end_time:
            topic_keyframes.append(kf)
    
    return topic_keyframes


def calculate_keyframe_distribution(topic_duration: float, min_keyframes: int = 10, max_keyframes: int = 40) -> int:
    """
    Calculate how many keyframes to use based on topic duration.
    
    Args:
        topic_duration: Duration of topic in seconds
        min_keyframes: Minimum number of keyframes
        max_keyframes: Maximum number of keyframes
        
    Returns:
        Target number of keyframes
    """
    # Base calculation: 1 keyframe per 2 seconds, with limits
    target_keyframes = int(topic_duration / 2)
    
    # Apply limits
    target_keyframes = max(min_keyframes, min(max_keyframes, target_keyframes))
    
    return target_keyframes
